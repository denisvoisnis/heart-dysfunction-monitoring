class DigitalFilter:
    def __init__(self):
        self.xIIR = None
        self.yIIR = None
        self.LIIRb = 0
        self.LIIRa = 0
        self.flagIIR = True

    def IIRFilter(self, input_val, b_coeff, a_coeff):
        if self.flagIIR:
            self.xIIR = [0.0] * len(b_coeff)
            self.yIIR = [0.0] * len(a_coeff)
            self.LIIRb = len(b_coeff) - 1
            self.LIIRa = len(a_coeff) - 1
            self.flagIIR = False

        # Calculate new output:
        output_iir = input_val * b_coeff[0]
        for i in range(self.LIIRb):
            output_iir += self.xIIR[i] * b_coeff[i + 1]
        for i in range(self.LIIRa):
            output_iir -= self.yIIR[i] * a_coeff[i + 1]

        # Update:
        for i in range(self.LIIRa, 0, -1):
            self.yIIR[i] = self.yIIR[i - 1]
        self.yIIR[0] = output_iir

        for i in range(self.LIIRb, 0, -1):
            self.xIIR[i] = self.xIIR[i - 1]
        self.xIIR[0] = input_val

        return output_iir