import torch


def reduce_by_key(values: torch.Tensor, keys: torch.Tensor, op: str = "sum") -> torch.Tensor:
    if op == "sum":
        op = 0
    elif op == "min":
        op = 1
    elif op == "max":
        op = 2
    else:
        raise NotImplementedError(op)

    return torch.ops.sparse_ops.reduce_by_key(values, keys, op)


def main():
    device = torch.device("cuda")

    points = torch.as_tensor(
        [
            [0.6, 0.7, 0.9],
            [0.2, 0.1, 0.3],
            [0.7, 0.6, 0.8],
            [0.1, 0.6, 0.3],
            [0.7, 0.7, 0.1],
        ],
        dtype=torch.float32,
        device=device,
    )
    batch_indices = torch.as_tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)

    results = reduce_by_key(
        points,
        keys=batch_indices,
    )
    print("results", results)


if __name__ == "__main__":
    main()
