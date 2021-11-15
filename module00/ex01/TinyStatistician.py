import math,sys,os

class TinyStatisticain():

    def __init__(self):
        pass

    def list_validator(self, l):
        if isinstance(l, list):
            if len(l) == 0:
                return False
            for val in l:
                if not isinstance(val, int) and not isinstance(val, float):
                    return False
                else:
                    val = float(val)
            return l
        else:
            return False

    def list_sort(self, l):
        i = 0
        while i < len(l):
            j = i
            while j < len(l):
                if l[j] < l[i]:
                    swap = l[i]
                    l[i] = l[j]
                    l[j] = swap
                j += 1
            i += 1
        return l

    def mean(self, x):
        l = self.list_validator(x)
        if l == False:
            mean = None
        else:
            x_sum = 0
            i = 0
            for val in l:
                x_sum += val
                i += 1
            if i != 0:
                mean = float(x_sum / i)
            else:
                mean = None
        print(mean)
        return mean

    def median(self, x):
        l = self.list_validator(x)
        if l == False:
            median = None
        else:
            l = self.list_sort(l)
            index = int(len(l) / 2)
            if len(l) % 2 != 0:
                median = float(l[index - 1])
            else:
                median = float((l[index] + l[index - 1]) / 2)
        print(median)
        return median

    def quartile(self, x, percentile):
        l = self.list_validator(x)
        if l == False:
            quartile = None
        else:
            l = self.list_sort(l)
            if percentile == 2 or percentile == 50:
                quartile = self.median(l)
            elif percentile == 0:
                quartile = float(l[0])
            elif percentile == 4 or percentile == 100:
                quartile = float(l[len(l)])
            elif percentile == 1 or percentile == 25:
                index = float((len(l) + 3) / 4)
                if len(l) % 2 != 0:
                    quartile = float(l[int(index) - 1])
                else:
                    i = int(index)
                    w1 = index - i
                    w2 = 1 - w1
                    quartile = float(l[i]) * w1 + float(l[i - 1]) * w2
            elif percentile == 3 or percentile == 75:
                index = (3 * len(l) + 1) / 4
                if len(l) % 2 != 0:
                    quartile = float(l[int(index) - 1])
                else:
                    i = int(index)
                    w1 = index - i
                    w2 = 1 - w1
                    quartile = float(l[i]) * w1 + float(l[i - 1]) * w2
            else:
                quartile = None
        print(quartile)
        return quartile

    def var(self, x, call=0):
        sys.stdout = open(os.devnull, 'w')
        if self.mean(x) != None:
            mean = self.mean(x)
            v1 = 0
            i = 0
            for value in x:
                v1 += float((value - mean) * (value - mean))
                i += 1
            var = v1 / i if i != 0 else None
        else:
            var = None
        if call == 0:
            sys.stdout = sys.__stdout__
        print(var)
        return var

    def std(self, x):
        sys.stdout = open(os.devnull, 'w')
        if self.var(x, 1) != None:
            std = float(math.sqrt(self.var(x, 1)))
        else:
            std = None
        sys.stdout = sys.__stdout__
        print(std)
        return std

#shirt = 'white' if game_type == 'home' else 'green'

tstat = TinyStatisticain()
a = [1, 42, 300, 10, 59]
b = [1, 2, 3, 4]
q1 = [1, 11, 15, 19, 20, 24, 28, 34, 37, 47, 50, 61, 68]
q2 = [1, 11, 15, 19, 20, 24, 28, 34, 37, 47, 50, 61]
tstat.mean(a)
tstat.median(a)
tstat.median(b)
tstat.quartile(a, 75)
tstat.quartile(b, 6)
tstat.quartile(q1, 3)
tstat.quartile(q2, 75)
tstat.var(a)
tstat.std(a)
