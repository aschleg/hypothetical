from hypothetical._lib import build_des_mat
from hypothetical.aov.anova import AnovaOneWay


class KruskalWallis(AnovaOneWay):

    def __init__(self, group=None, *args):
        super().__init__(self)

        if group is not None:
            self.group = group

        self.design_matrix = build_des_mat(group, *args)

    def _tie_correction(self):
        pass

    def _h_value(self):
        pass

    def _p_value(self):
        pass

    def _t_value(self):
        pass