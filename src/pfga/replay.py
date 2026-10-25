"""Event replay; policies never receive the observed worker as an allocation feature."""

import heapq
import math
from functools import lru_cache
import torch
from pfga.data import MonthlyBatch, _torch_load


def haversine(a, b):
    a, b = torch.deg2rad(a), torch.deg2rad(b)
    d = b[None, :, :] - a[:, None, :]
    h = (
        torch.sin(d[..., 0] / 2) ** 2
        + torch.cos(a[:, None, 0]) * torch.cos(b[None, :, 0]) * torch.sin(d[..., 1] / 2) ** 2
    )
    return 6371.0 * 2 * torch.asin(h.clamp(0, 1).sqrt())


class StreamData:
    def __init__(self, path, config):
        data = _torch_load(path)
        self.common, self.tasks, self.split = data["common"], data["tasks"], data["split"]
        if not self.tasks or any(a["time"] > b["time"] for a, b in zip(self.tasks, self.tasks[1:])):
            raise ValueError("Nonempty chronologically sorted stream required")
        self.config = config
        self.num_workers = self.common["worker_features"].shape[0]
        self.worker_dim = self.common["worker_features"].shape[1]
        self.task_dim = self.common["num_categories"] + 4
        self.task_by_id = {t["id"]: t for t in self.tasks}
        self.reference = lru_cache(maxsize=4096)(self._reference_for)

    def window(self, time):
        c = self.common
        starts = c["centers"] - c["window_hours"] / 2
        day = torch.floor((time - starts) / 24)
        opens = day * 24 + starts
        return (
            day.long(),
            (time >= opens) & (time < opens + c["window_hours"]),
            opens + c["window_hours"],
        )

    def _reference_for(self, task_id):
        """Bounded cached reference samples, deterministic per release identity."""
        c = self.common
        task = self.task_by_id[task_id]
        gen = torch.Generator().manual_seed(self.config.data.sampling_seed + task_id)
        _, online, ends = self.window(task["time"])
        distance = haversine(
            c["initial_locations"], torch.tensor([task["location"]], dtype=torch.float64)
        )[:, 0]
        ok = online & (
            distance / c["speed_kmh"]
            <= torch.minimum(ends - task["time"], torch.full_like(ends, c["deadline_hours"]))
        )
        eligible = ok.nonzero().flatten()
        pos = task["worker"]
        positive = eligible[eligible == pos]
        negatives = eligible[eligible != pos]
        # No invented positive when recorded user is infeasible under the simulation.
        if positive.numel():
            negatives = negatives[
                torch.randperm(len(negatives), generator=gen)[: self.config.data.negative_ratio]
            ]
            sampled = torch.cat((positive, negatives))
        else:
            sampled = torch.empty(0, dtype=torch.long)
        return sampled, eligible

    def replay(self):
        return Replay(self)


class Replay:
    def __init__(self, data):
        self.data = data
        self.c = data.common
        self.m = data.num_workers
        self.locations = self.c["initial_locations"].clone()
        self.next_time = torch.full((self.m,), -math.inf, dtype=torch.float64)
        self.window_ids = torch.full((self.m,), -(2**60), dtype=torch.long)
        self.remaining = self.c["capacities"].clone()
        self.pending = {}
        self.events = []
        self.counter = 0
        self.batch = None
        for task in data.tasks:
            self.push(task["time"], "release", task)
            self.push(task["time"] + self.c["deadline_hours"], "expiry", task["id"])
        # Worker window transitions provide retry opportunities even with no task release.
        first = data.tasks[0]["time"]
        self.horizon = data.tasks[-1]["time"] + self.c["deadline_hours"]
        for center in torch.unique(self.c["centers"]).tolist():
            for offset in (-self.c["window_hours"] / 2, self.c["window_hours"] / 2):
                start = center + offset
                t = math.ceil((first - start) / 24) * 24 + start
                if t <= self.horizon:
                    self.push(t, "window", None)
        self.new_tasks = []

    def push(self, t, kind, value):
        heapq.heappush(self.events, (t, self.counter, kind, value))
        self.counter += 1

    def __iter__(self):
        return self

    def __next__(self):
        while self.events:
            t = self.events[0][0]
            self.time = t
            self.new_tasks = []
            while self.events and abs(self.events[0][0] - t) < 1e-9:
                _, _, kind, value = heapq.heappop(self.events)
                if kind == "window" and t + 24 <= self.horizon:
                    self.push(t + 24, "window", None)
                if kind == "release":
                    self.pending[value["id"]] = value
                    self.new_tasks.append(value)
                if kind == "expiry":
                    self.pending.pop(value, None)
            if not self.pending:
                continue
            win, online, end = self.data.window(t)
            changed = win != self.window_ids
            self.remaining[changed] = self.c["capacities"][changed]
            self.window_ids = win
            cap = self.remaining.clone()
            cap[~online | (self.next_time > t)] = 0
            tasks = list(self.pending.values())
            self.local_tasks = tasks
            n = len(tasks)
            loc = torch.tensor([x["location"] for x in tasks], dtype=torch.float64)
            dist = haversine(self.locations, loc).float()
            deadline = torch.tensor(
                [x["time"] + self.c["deadline_hours"] for x in tasks], dtype=torch.float64
            )
            budget = torch.minimum(deadline[None, :], end[:, None]) - t
            feasible = (cap[:, None] > 0) & (budget >= 0) & (dist / self.c["speed_kmh"] <= budget)
            edges = feasible.nonzero().T.contiguous()
            cats = torch.tensor([x["category"] for x in tasks])
            rewards = torch.tensor([x["reward"] for x in tasks])
            labels = torch.tensor([[x["worker"] for x in tasks], list(range(n))])
            ref = []
            opportunity = torch.zeros(self.m)
            new_ids = {x["id"] for x in self.new_tasks}
            for j, task in enumerate(tasks):
                if task["id"] in new_ids:
                    sampled, eligible = self.data.reference(task["id"])
                    ref.extend((i, j) for i in sampled.tolist())
                    opportunity[eligible] += 1
            ref = torch.tensor(ref, dtype=torch.long).reshape(-1, 2).T.contiguous()
            hour = torch.tensor([x["time"] % 24 for x in tasks])
            tf = torch.cat(
                (
                    torch.nn.functional.one_hot(cats, self.c["num_categories"]).float(),
                    loc.float() / torch.tensor([90.0, 180.0]),
                    torch.sin(hour[:, None] * math.pi / 12),
                    torch.cos(hour[:, None] * math.pi / 12),
                ),
                1,
            )
            wf = self.c["worker_features"].clone()
            c = self.c["num_categories"]
            wf[:, c : c + 2] = self.locations.float() / torch.tensor([90.0, 180.0])
            self.batch = MonthlyBatch(
                period=str(t),
                worker_features=wf,
                task_features=tf,
                positive_edge_index=labels,
                social_edge_index=self.c["social_edge_index"],
                feasible_edge_index=edges,
                benefit_matrix=self.c["affinity"][:, cats] * rewards[None, :],
                cost_matrix=dist,
                preference_matrix=self.c["retrieval_profile"][:, cats],
                slack_matrix=(
                    1 - dist / (self.c["speed_kmh"] * budget.float().clamp_min(0) + 1e-8)
                ).clamp(0, 1),
                worker_ids=torch.arange(self.m),
                worker_capacity=cap,
                task_rewards=rewards,
                supervision_edge_index=ref,
                reference_opportunities=opportunity,
                distance_scale_km=self.data.config.evaluation.distance_scale_km,
            )
            return self.batch
        raise StopIteration

    def prediction_batch(self):
        """Reference covariates independent of policy trajectory, labels only for scoring."""
        from dataclasses import replace

        return replace(self.batch, worker_features=self.c["worker_features"])

    def accept(self, i, j):
        task = self.local_tasks[j]
        if (
            task["id"] not in self.pending
            or self.remaining[i] <= 0
            or self.next_time[i] > self.time
        ):
            return False
        _, online, end = self.data.window(self.time)
        distance = float(
            haversine(
                self.locations[i : i + 1], torch.tensor([task["location"]], dtype=torch.float64)
            )[0, 0]
        )
        finish = self.time + distance / self.c["speed_kmh"]
        if not online[i] or finish > min(task["time"] + self.c["deadline_hours"], float(end[i])):
            return False
        self.remaining[i] -= 1
        self.locations[i] = torch.tensor(task["location"], dtype=torch.float64)
        # One simultaneous job; zero-distance tasks become available at a later event.
        self.next_time[i] = finish + 1e-7
        self.push(finish + 1e-7, "complete", None)
        self.pending.pop(task["id"])
        return True
