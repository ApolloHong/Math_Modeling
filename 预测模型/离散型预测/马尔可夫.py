from numpy import zeros, sum, array, ones


class DTMC:
    def __init__(self, size, state_names=None, init_state=None, eps=0.001, max_iter=1e5):
        if state_names is not None:
            self.state_names = state_names
        else:
            self.state_names = []
            for _ in range(size):
                self.state_names.append('state_' + str(_))
        self.size = size
        if init_state is not None:
            self.p_i = array(init_state)
        else:
            self.p_i = ones((1, self.size)) / size
        self.trans = zeros((self.size, self.size))
        self.max_iter = max_iter
        self.eps = eps
        self.result = None

    def reset_state(self, s):
        self.p_i = array(s).reshape((1, self.size))

    def fit(self, trans):
        trans = array(trans)
        res = self.p_i.copy()
        for _ in range(int(self.max_iter)):
            print('\riter', _, end='')
            rec = res.copy()
            res = res.dot(trans)
            if sum(abs(res - rec)) < self.eps:
                print('\rDTMC: Converge after', _, 'iteration(s).')
                break
            if _ == self.max_iter - 1:
                if self.eps > 0:
                    print('\rWarning: Did not converge! Epsilon too small or max_iter not enough. Final change:',
                          sum(abs(res - rec)))
                else:
                    print('\rYou\'ve disabled Epsilon. Final change:', sum(abs(res - rec)))
        self.result = res

    def predict(self, index=None):
        if self.result is None:
            print('Fit the model first.')
            return
        if index is not None:
            if index >= self.size:
                raise IndexError
            print('\rDTMC: predict', self.state_names[index], 100 * self.result[0, index], '%')
            return self.result[0, index]
        else:
            print('\rDTMC:\n', '\b' + self.state_names[0], 100 * self.result[0, 0], '%')
            for _ in range(1, self.size):
                print(self.state_names[_], 100 * self.result[0, _], '%')
            return self.result


# 该函数包含了所有合法使用方法
def sampleDTMC():
    mc = DTMC(4)

    mc.predict()  # Will give warning
    print('\n')

    trans = [[0, 0.1, 0.4, 0.5], [0.3, 0, 0, 0.7], [0, 0, 0, 1], [0, 0.4, 0.6, 0]]
    mc.fit(trans)
    mc.predict()
    print('\n')

    mc.reset_state([0.1, 0.2, 0.3, 0.4])  # or define as: mc = DTMC(4, init_state=[0.1, 0.2, 0.3, 0.4])
    mc.fit(trans)
    mc.predict()
    print('\n')

    mc.eps = 1e-10  # or define as: mc = DTMC(4, init_state=[0.1, 0.2, 0.3, 0.4], eps=1e-10)
    mc.fit(trans)
    mc.predict()
    print('\n')

    mc.state_names = ['sunny', 'rainy', 'foggy', 'cloudy']
    # or define as: mc = DTMC(..., state_names=['sunny', 'rainy', 'foggy', 'cloudy'])
    mc.predict(2)
    print('\n')

    mc.eps = -1  # not allow break
    # or define as: mc = DTMC(..., eps=-1)
    mc.fit(trans)
    mc.predict()
    print('\n')

    mc.eps = 0.001
    mc.max_iter = 10
    # or define as: mc = DTMC(..., max_iter=10)
    mc.fit(trans)
    mc.predict()
    print('\n')


sampleDTMC()

model = DTMC(3, state_names=['0umbrella', '1umbrella', '2umbrellas'])
model.fit([[0, 0, 1], [0, 0.3, 0.7], [0.3, 0.7, 0]])
res = model.predict(0)
cal_pro = res * 0.7
print('calculation 传染病模型:', cal_pro)

rain = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
total = 1000000  # how many times the man travel btw home and office
wet = 0
pos = 0
umb = [1, 1]  # you can try [0,2] or [2,0]
for _ in range(total):
    shuffle(rain)
    is_rain = rain[0]
    if is_rain:
        if umb[pos] == 0:
            wet += 1
        else:
            umb[pos] -= 1
            umb[1 - pos] += 1
    pos = 1 - pos
sim_pro = wet / total
print('simulation 传染病模型:', sim_pro)
