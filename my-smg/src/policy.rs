use crate::worker::{HealthStatus, Worker};

pub fn healthy_worker_indices(workers: &[Worker]) -> Vec<usize> {
    let mut indices = Vec::new();

    for (idx, worker) in workers.iter().enumerate() {
        if worker.status() == &HealthStatus::Healthy {
            indices.push(idx);
        }
    }

    indices
}

#[derive(Debug)]
pub struct FirstHealthy;

impl FirstHealthy {
    pub fn new() -> Self {
        Self
    }
    pub fn select(&self, workers: &[Worker]) -> Option<usize> {
        //copied() 能使用，是因为 usize 实现了 Copy
        healthy_worker_indices(workers).first().copied()
    }
}

#[derive(Debug)]
pub struct RoundRobin {
    next: usize,
}

impl RoundRobin {
    pub fn new() -> Self {
        Self { next: 0 }
    }

    //这里使用&mut self 因为需要对self.next做更新,但是Worker不修改，所以&
    pub fn select(&mut self, workers: &[Worker]) -> Option<usize> {
        let healthy_indices = healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }
        let position = self.next % healthy_indices.len();
        let selected_index = healthy_indices[position];
        self.next = (position + 1) % healthy_indices.len();
        Some(selected_index)
    }
}

#[cfg(test)]
mod tests {
    use super::{FirstHealthy, RoundRobin, healthy_worker_indices};
    use crate::worker::{HealthStatus, Worker};

    #[test]
    fn first_healthy_selects_first_available_worker() {
        let mut worker_a = Worker::new(String::from("worker-a"));
        let worker_b = Worker::new(String::from("worker-b"));
        let worker_c = Worker::new(String::from("worker-c"));

        worker_a.set_status(HealthStatus::Unhealthy);

        let workers = vec![worker_a, worker_b, worker_c];
        let policy = FirstHealthy::new();

        assert_eq!(policy.select(&workers), Some(1));
    }

    #[test]
    fn first_healthy_returns_none_when_none_are_available() {
        let mut worker = Worker::new(String::from("worker-a"));
        worker.set_status(HealthStatus::Unhealthy);

        let workers = vec![worker];
        let policy = FirstHealthy::new();

        assert_eq!(policy.select(&workers), None);
    }

    #[test]
    fn round_robin_cycles_through_healthy_workers() {
        let workers = vec![
            Worker::new(String::from("worker-a")),
            Worker::new(String::from("worker-b")),
            Worker::new(String::from("worker-c")),
        ];

        let mut policy = RoundRobin::new();

        assert_eq!(policy.select(&workers), Some(0));
        assert_eq!(policy.select(&workers), Some(1));
        assert_eq!(policy.select(&workers), Some(2));
        assert_eq!(policy.select(&workers), Some(0));
    }

    #[test]
    fn empty_worker_list_has_no_healthy_indices() {
        let workers: Vec<Worker> = Vec::new();

        assert_eq!(healthy_worker_indices(&workers), Vec::<usize>::new());
    }

    #[test]
    fn all_unhealthy_workers_have_no_healthy_indices() {
        let mut worker_a = Worker::new(String::from("worker-a"));
        let mut worker_b = Worker::new(String::from("worker-b"));

        worker_a.set_status(HealthStatus::Unhealthy);
        worker_b.set_status(HealthStatus::Unhealthy);

        let workers = vec![worker_a, worker_b];

        assert_eq!(healthy_worker_indices(&workers), Vec::<usize>::new());
    }

    #[test]
    fn resturs_original_indices_of_healthy_workers() {
        let worker_a = Worker::new(String::from("worker-a"));
        let mut worker_b = Worker::new(String::from("worker-b"));
        worker_b.set_status(HealthStatus::Unhealthy);

        let worker_c = Worker::new(String::from("worker-c"));

        let workers = vec![worker_a, worker_b, worker_c];
        assert_eq!(healthy_worker_indices(&workers), vec![0, 2]);
    }

    #[test]
    fn round_robin_returns_none_for_empty_list() {
        let workers: Vec<Worker> = Vec::new();
        let mut policy = RoundRobin::new();

        assert_eq!(policy.select(&workers), None);
    }

    #[test]
    fn round_robin_repeats_single_healthy_worker() {
        let workers = vec![Worker::new(String::from("worker-a"))];
        let mut policy = RoundRobin::new();

        assert_eq!(policy.select(&workers), Some(0));
        assert_eq!(policy.select(&workers), Some(0));
        assert_eq!(policy.select(&workers), Some(0));
    }

    #[test]
    fn round_robin_skips_unhealthy_workers() {
        let worker_a = Worker::new(String::from("worker-a"));
        let mut worker_b = Worker::new(String::from("worker-b"));
        let worker_c = Worker::new(String::from("worker-c"));

        worker_b.set_status(HealthStatus::Unhealthy);

        let workers = vec![worker_a, worker_b, worker_c];
        let mut policy = RoundRobin::new();

        assert_eq!(policy.select(&workers), Some(0));
        assert_eq!(policy.select(&workers), Some(2));
        assert_eq!(policy.select(&workers), Some(0));
        assert_eq!(policy.select(&workers), Some(2));
    }

    #[test]
    fn round_robin_returns_none_when_all_workers_are_unhealthy() {
        let mut worker_a = Worker::new(String::from("worker-a"));
        let mut worker_b = Worker::new(String::from("worker-b"));

        worker_a.set_status(HealthStatus::Unhealthy);
        worker_b.set_status(HealthStatus::Unhealthy);

        let workers = vec![worker_a, worker_b];
        let mut policy = RoundRobin::new();

        assert_eq!(policy.select(&workers), None);
    }

    #[test]
    fn round_robin_responds_to_worker_health_changes() {
        let mut workers = vec![
            Worker::new(String::from("worker-a")),
            Worker::new(String::from("worker-b")),
            Worker::new(String::from("worker-c")),
        ];

        let mut policy = RoundRobin::new();
        assert_eq!(policy.select(&workers), Some(0));

        workers[1].set_status(HealthStatus::Unhealthy);

        for _ in 0..4 {
            assert_ne!(policy.select(&workers), Some(1));
        }
        workers[1].set_status(HealthStatus::Healthy);

        let next_three = [
            policy.select(&workers),
            policy.select(&workers),
            policy.select(&workers),
        ];

        assert!(next_three.contains(&Some(1)));
    }
}
