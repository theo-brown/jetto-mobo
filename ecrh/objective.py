import numpy as np
from netCDF4 import Dataset


def f1(profiles: Dataset, timetraces: Dataset):
    """q(0) - min(q)"""
    q = profiles["Q"][-1].data
    return q[0] - np.min(q)


def f2(profiles: Dataset, timetraces: Dataset, epsilon=0.3):
    """|| min(q) - (2 + epsilon) ||"""
    q = profiles["Q"][-1].data
    return np.abs(np.min(q) - (2 + epsilon))


def f3(profiles: Dataset, timetraces: Dataset):
    """argmin(q)"""
    q = profiles["Q"][-1].data
    return np.argmin(q)


def f4(profiles: Dataset, timetraces: Dataset, epsilon=0.03):
    """li - (0.25 + epsilon)"""
    li = timetraces["LI"][-1].data
    return np.abs(li - (0.25 + epsilon))


def f5(profiles: Dataset, timetraces: Dataset):
    """Fraction of locations where q is non-increasing
    Calculated by summing q[i] <= q[i-1] and dividing by len(q)"""
    q = profiles["Q"][-1].data
    return np.sum(q[1:] <= q[:-1]) / len(q)


def f6(profiles: Dataset, timetraces: Dataset):
    """Fraction of locations where dq/dr is non-increasing
    Calculated by summing dq[i] <= dq[i-1] and dividing by len(dq)"""
    q = profiles["Q"][-1].data
    dq = np.gradient(q)
    return np.sum(dq[1:] <= dq[:-1]) / len(q)


def f7(profiles: Dataset, timetraces: Dataset, value=3):
    """-dq at first point where q>=value and r >= argmin(q)"""
    q = profiles["Q"][-1].data
    dq = np.gradient(q)
    condition_1 = q >= value
    condition_2 = np.arange(len(q)) >= np.argmin(q)
    # Get index of element where both conditions are met
    i = np.where(condition_1 & condition_2)[0][0]
    return -dq[i]


def f8(profiles: Dataset, timetraces: Dataset):
    """-dq at first point where q>=4 and r >= argmin(q)"""
    return f7(profiles, timetraces, value=4)


def combined_cost_function(profiles: Dataset, timetraces: Dataset):
    return (
        0.5 * f1(profiles, timetraces)
        + 5 * f2(profiles, timetraces)
        + 6 * f3(profiles, timetraces)
        + 10 * f5(profiles, timetraces)
        + 10 * f6(profiles, timetraces)
        + 1 * f7(profiles, timetraces)
        + 2 * f8(profiles, timetraces)
    )


if __name__ == "__main__":
    from jetto_tools.results import JettoResults

    results = JettoResults(
        path="/home/theo/Documents/cambridge/iib-project/data/jetto/benchmark"
    )
    profiles = results.load_profiles()
    timetraces = results.load_timetraces()

    print(
        "Benchmark combined cost function: "
        f"{combined_cost_function(profiles, timetraces):3e}"
    )
    print(
        "Benchmark individual cost functions: \n"
        f"\t f1: {f1(profiles, timetraces):3e} \n"
        f"\t f2: {f2(profiles, timetraces):3e} \n"
        f"\t f3: {f3(profiles, timetraces):3e} \n"
        f"\t f4: {f4(profiles, timetraces):3e} \n"
        f"\t f5: {f5(profiles, timetraces):3e} \n"
        f"\t f6: {f6(profiles, timetraces):3e} \n"
        f"\t f7: {f7(profiles, timetraces):3e} \n"
        f"\t f8: {f8(profiles, timetraces):3e} \n"
    )
