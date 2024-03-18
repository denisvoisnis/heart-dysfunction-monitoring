class Resample:
    def byte_to_short(self, data, length, reduce_stereo):
        if reduce_stereo:
            short_data = []
            for i in range(0, len(data), 4):
                val1 = (data[i + 1] << 8) | data[i]
                val2 = (data[i + 3] << 8) | data[i + 2]
                short_data.append(((val1 + val2) // 2) & 0xFFFF)  # Ensure 16-bit overflow
            return short_data
        else:
            return [((data[i + 1] << 8) | data[i]) & 0xFFFF for i in range(0, len(data), 2)]

    def short_to_byte(self, data, length, expand_mono):
        byte_data = bytearray()

        if expand_mono:
            for val in data:
                byte_data.extend([val & 0xFF, (val >> 8) & 0xFF, val & 0xFF, (val >> 8) & 0xFF])
        else:
            for val in data:
                byte_data.extend([val & 0xFF, (val >> 8) & 0xFF])

        return byte_data

    def convert(self, data, length, in_stereo, out_stereo, in_frequency, out_frequency):
        if in_stereo == out_stereo and in_frequency == out_frequency:
            return self.trim_array(data, length)

        converted = self.byte_to_short(data, length, in_stereo and not out_stereo)
        converted = self.resample(converted, len(converted), out_stereo, in_frequency, out_frequency)
        return self.short_to_byte(converted, len(converted), not in_stereo and out_stereo)

    def resample(self, data, length, stereo, in_frequency, out_frequency):
        if in_frequency < out_frequency:
            return self.upsample(data, length, stereo, in_frequency, out_frequency)
        if in_frequency > out_frequency:
            return self.downsample(data, length, stereo, in_frequency, out_frequency)
        return self.trim_array(data, length)

    def upsample(self, data, length, stereo, in_frequency, out_frequency):
        if in_frequency == out_frequency:
            return self.trim_array(data, length)

        scale = in_frequency / out_frequency
        pos = 0.0
        output = []

        if not stereo:
            for _ in range(int(length / scale)):
                in_pos = int(pos)
                proportion = pos - in_pos

                if in_pos >= length - 1:
                    in_pos = length - 2
                    proportion = 1.0

                output.append(round(data[in_pos] * (1.0 - proportion) + data[in_pos + 1] * proportion))
                pos += scale
        else:
            for _ in range(int((length / 2) / scale)):
                in_pos = int(pos)
                proportion = pos - in_pos

                in_real_pos = in_pos * 2
                if in_real_pos >= length - 3:
                    in_real_pos = length - 4
                    proportion = 1.0

                output.extend([
                    round(data[in_real_pos] * (1.0 - proportion) + data[in_real_pos + 2] * proportion),
                    round(data[in_real_pos + 1] * (1.0 - proportion) + data[in_real_pos + 3] * proportion)
                ])
                pos += scale

        return output

    def downsample(self, data, length, stereo, in_frequency, out_frequency):
        if in_frequency == out_frequency:
            return self.trim_array(data, length)

        scale = out_frequency / in_frequency
        output = []
        pos = 0.0
        out_pos = 0

        if not stereo:
            sum_val = 0.0
            output = [0] * int(length * scale)
            in_pos = 0
            while out_pos < len(output):
                this_val = data[in_pos]
                in_pos += 1
                next_pos = pos + scale
                if next_pos >= 1.0:
                    sum_val += this_val * (1.0 - pos)
                    output[out_pos] = round(sum_val)
                    out_pos += 1
                    next_pos -= 1.0
                    sum_val = next_pos * this_val
                else:
                    sum_val += scale * this_val
                pos = next_pos

                if in_pos >= length and out_pos < len(output):
                    output[out_pos] = round(sum_val / pos)
                    out_pos += 1
        else:
            sum1 = 0.0
            sum2 = 0.0
            output = [0] * (2 * int(length / 2 * scale))
            in_pos = 0
            while out_pos < len(output):
                this_val1 = data[in_pos]
                this_val2 = data[in_pos + 1]
                next_pos = pos + scale
                if next_pos >= 1.0:
                    sum1 += this_val1 * (1.0 - pos)
                    sum2 += this_val2 * (1.0 - pos)
                    output[out_pos] = round(sum1)
                    output[out_pos + 1] = round(sum2)
                    next_pos -= 1.0
                    sum1 = next_pos * this_val1
                    sum2 = next_pos * this_val2
                else:
                    sum1 += scale * this_val1
                    sum2 += scale * this_val2
                pos = next_pos

                if in_pos >= length and out_pos < len(output):
                    output[out_pos] = round(sum1 / pos)
                    output[out_pos + 1] = round(sum2 / pos)
                    out_pos += 2

        return output

    def trim_array(self, data, length):
        if len(data) == length:
            return data
        else:
            return data[:length]