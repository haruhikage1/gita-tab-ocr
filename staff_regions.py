from gtrs.model import TabMultiStaff, TabStaff


def get_center_min_max_y(staff: TabStaff) -> tuple[float, float]:
    return (staff.min_y, staff.max_y)


class TabStaffRegions:
    def __init__(self, staffs: list[TabMultiStaff]) -> None:
        self.centers = [get_center_min_max_y(s) for ms in staffs for s in ms.staffs]

    def get_start_of_closest_staff_above(self, y: float) -> float:
        staffs_above = [c[1] for c in self.centers if c[0] < y]
        if len(staffs_above) == 0:
            return 0
        return max(staffs_above)

    def get_start_of_closest_staff_below(self, y: float) -> float:
        staffs_below = [c[0] for c in self.centers if c[1] > y]
        if len(staffs_below) == 0:
            return 1e12
        return min(staffs_below)