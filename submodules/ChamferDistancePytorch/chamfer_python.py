import torch


def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    P = rx.t() + ry - 2 * zz
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


def batched_pairwise_dist(a, b):
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def distChamfer(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    P = batched_pairwise_dist(a, b)
    return torch.min(P, 2)[0].float(), torch.min(P, 1)[0].float(), torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()


def distChamfer_a2b(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    P = batched_pairwise_dist(a, b)
    return torch.min(P, 2)[0].float(), torch.min(P, 2)[1].int()


def distChamfer_a2b_wDist(a, b, retain_dim=10):
    """
    :param a: Pointclouds [batch_size, num_points_a, dim] - First point cloud
    :param b: Pointclouds [batch_size, num_points_b, dim] - Second point cloud 
    :param retain_dim: int - Number of closest points to retain
    :return:
        - min_dists: [batch_size, num_points_a] - Distance to closest point in b for each point in a
        - min_indices: [batch_size, num_points_a] - Index of closest point in b for each point in a
        - topk_dists: [batch_size, num_points_a, retain_dim] - Distances to k closest points in b
        - topk_indices: [batch_size, num_points_a, retain_dim] - Indices of k closest points in b
    """
    P = batched_pairwise_dist(a, b)

    # min_dists: [batch_size, num_points_a] - Distance to closest point in b for each point in a
    # min_indices: [batch_size, num_points_a] - Index of closest point in b for each point in a
    min_dists, min_indices = torch.min(P, dim=2)

    # topk_dists: [batch_size, num_points_a, k] - k smallest distances for each point in a
    # topk_indices: [batch_size, num_points_a, k] - indices of k closest points in b for each point in a
    topk_dists, topk_indices = torch.topk(P, k=retain_dim, dim=2, largest=False)

    return min_dists.float(), min_indices, topk_dists.float(), topk_indices


def distChamfer_a2b_wDist2(a, b, retain_dim=10):
    """
    :param a: Pointclouds [batch_size, num_points_a, dim] - First point cloud
    :param b: Pointclouds [batch_size, num_points_b, dim] - Second point cloud 
    :param retain_dim: int - Number of closest points to retain
    :return:
        - min_dists: [batch_size, num_points_a] - Distance to closest point in b for each point in a
        - min_indices: [batch_size, num_points_a] - Index of closest point in b for each point in a
        - topk_dists: [batch_size, num_points_a, retain_dim] - Distances to k closest points in b
        - topk_indices: [batch_size, num_points_a, retain_dim] - Indices of k closest points in b
    """
    P = batched_pairwise_dist(a, b)

    # min_dists: [batch_size, num_points_a] - Distance to closest point in b for each point in a
    # min_indices: [batch_size, num_points_a] - Index of closest point in b for each point in a
    min_indices = torch.min(P, dim=2)[1]

    # topk_dists: [batch_size, num_points_a, k] - k smallest distances for each point in a
    # topk_indices: [batch_size, num_points_a, k] - indices of k closest points in b for each point in a
    topk_dists, topk_indices = torch.topk(P, k=retain_dim, dim=2, largest=False)

    return min_indices, topk_dists.float(), topk_indices

