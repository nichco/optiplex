import philote_mdo.general as pmdo

class SubProblem(pmdo.ExplicitDiscipline):

    def setup(self):
        self.add_input("x")
        self.add_output("x_i_star")

    def compute(self, inputs, outputs):
        x = inputs["x"]

        outputs["x_i_star"] = solution(x)