class DigitalAdaptiveFilter:
    def __init__(self):
        self.a = 0.0001
        self.i = 0
        self.w = None
        self.x = None
        self.e = 0.0
        self.y = 0.0
        self.norm = 0.0
        self.flag = True

    def NLMS(self, mu, M, d, ref):
        if self.flag:
            self.w = [0.0] * M
            self.x = [0.0] * M
            self.flag = False

        self.y = self.w[0] * ref

        for i in range(M - 1):
            self.y += self.w[i + 1] * self.x[i]

        self.norm = sum([xi * xi for xi in self.x]) ** 0.5

        self.e = d - self.y

        for i in range(M):
            self.w[i] += (mu / (self.norm ** 2 + self.a)) * (self.e * self.x[i])

        for i in range(M - 1, 0, -1):
            self.x[i] = self.x[i - 1]
        self.x[0] = ref

        return self.e