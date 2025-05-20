import sys
import numpy as np
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt

# A very small constant
# TINY = np.finfo(np.float64).tiny # 2e-308
EPS = np.finfo(np.float64).eps # 2e-16

# Set the seed, if needed
np.random.seed(hash("Ford, you are turning into a penguin, stop!") % 2**31)

DEFAULT_PHI = 0.5

# ---- Length and amount of runs

TIME_HORIZON = 1000
RUNS = 25

# ---- Feasibility and starting position

MIN_INV = 0
MIN_CASH = 0

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

INITIAL_STATE = dotdict({
	"price": 100,
	"taker": dotdict({"inv": 10, "cash": 1000}),
	"maker": dotdict({"inv": 10, "cash": 1000}),
})

# ---- State functions

def A(state):
	return min(state.maker.inv - MIN_INV, (state.taker.cash - MIN_CASH) / state.price)

def B(state):
	return min((state.maker.cash - MIN_CASH) / state.price, state.taker.inv - MIN_INV)

# ---- Misc

def sanity_check(state):
	assert state.price > 0, f"negative price ({state.price})"
	assert state.taker.inv > MIN_INV, f"taker inv too low ({state.taker.inv})"
	assert state.maker.inv > MIN_INV, f"maker inv too low ({state.maker.inv})"
	assert state.taker.cash > MIN_CASH, f"taker cash too low ({state.taker.cash})"
	assert state.maker.cash > MIN_CASH, f"maker cash too low ({state.maker.cash})"

def avg(xs):
	return np.cumsum(xs) / np.arange(1, len(xs) + 1)

def t(xs):
	return abs(min(xs[xs != 0], key=abs))

def to_unconstrained(x, a, b):
    # x = np.clip(x, a + EPS, b - EPS)
    return np.log((x - a) / (b - x))

def to_constrained(y, a, b):
    return a + (b - a) / (1 + np.exp(-y))

def dx_dy(y, a, b):
    s = 1 / (1 + np.exp(-y))
    return (b - a) * s * (1 - s)

# w => constrained space
# tilde w => unconstrained space
# w_half => extra gradient intermediate step
# w_one => updated point

def ascent(grad_f, w, par_min, par_max, learning_rate):
	w_one = w + learning_rate * grad_f(w)
	return np.clip(w_one, par_min + EPS, par_max - EPS)

def descent(grad_f, w, par_min, par_max, learning_rate):
	w_one = w - learning_rate * grad_f(w)
	return np.clip(w_one, par_min + EPS, par_max - EPS)

def ascent_in_unconstrained_space(grad_f, w, par_min, par_max, learning_rate):
	tilde_w = to_unconstrained(w, par_min, par_max)
	grad_wrt_tilde_w = grad_f(w) * dx_dy(tilde_w, par_min, par_max)
	tilde_w_one = tilde_w + learning_rate * grad_wrt_tilde_w 
	return to_constrained(tilde_w_one, par_min, par_max)

def descent_in_unconstrained_space(grad_f, w, par_min, par_max, learning_rate):
	tilde_w = to_unconstrained(w, par_min, par_max)
	grad_wrt_tilde_w = grad_f(w) * dx_dy(tilde_w, par_min, par_max)
	tilde_w_one = tilde_w - learning_rate * grad_wrt_tilde_w 
	return to_constrained(tilde_w_one, par_min, par_max)

def extra_ascent_in_unconstrained_space(grad_f, w, par_min, par_max, learning_rate):
	tilde_w = to_unconstrained(w, par_min, par_max)
	grad_wrt_tilde_w = grad_f(w) * dx_dy(tilde_w, par_min, par_max)

	tilde_w_half = tilde_w + learning_rate * grad_wrt_tilde_w
	w_half = to_constrained(tilde_w_half, par_min, par_max)
	grad_wrt_tilde_w_half = grad_f(w_half) * dx_dy(tilde_w_half, par_min, par_max)

	tilde_w_one = tilde_w + learning_rate * grad_wrt_tilde_w_half
	return to_constrained(tilde_w_one, par_min, par_max)

def extra_descent_in_unconstrained_space(grad_f, w, par_min, par_max, learning_rate):
	tilde_w = to_unconstrained(w, par_min, par_max)
	grad_wrt_tilde_w = grad_f(w) * dx_dy(tilde_w, par_min, par_max)

	tilde_w_half = tilde_w - learning_rate * grad_wrt_tilde_w
	w_half = to_constrained(tilde_w_half, par_min, par_max)
	grad_wrt_tilde_w_half = grad_f(w_half) * dx_dy(tilde_w_half, par_min, par_max)

	tilde_w_one = tilde_w - learning_rate * grad_wrt_tilde_w_half
	return to_constrained(tilde_w_one, par_min, par_max)

 # ---- Strategy

class Strategy:
	def __init__(self, kalpha, kbeta, valpha, vbeta, learning_rate=10**-1):
		self.phi = DEFAULT_PHI
		self.kalpha = kalpha
		self.kbeta = kbeta
		self.valpha = valpha
		self.vbeta = vbeta
		self.check()

		self.learning_rate = learning_rate

	def kalpha_grad(self, kalpha):
		# return self.valpha / (4 * kalpha * self.valpha + 3 * np.sqrt(2 * kalpha * self.valpha))
		# c = 2/3 * np.sqrt(2 * self.valpha)
		# return c / (2 * (np.sqrt(kalpha) - c * kalpha))

		# I'M IGNORING CONSTANTS IN ALL THE GRADIENTS
		return self.valpha / (1 + kalpha * self.valpha)
    
	def kbeta_grad(self, kbeta):
		# return self.vbeta / (4 * kbeta * self.vbeta - 3 * np.sqrt(2 * kbeta * self.vbeta))
		# c = 2/3 * np.sqrt(2 * self.vbeta)
		# return -c / (2 * (np.sqrt(kbeta) - c * kbeta))
		return -self.vbeta / (1 - kbeta * self.vbeta)

	def valpha_grad(self, valpha):
		# return self.valpha / (4 * self.kalpha * valpha + 3 * np.sqrt(2 * self.kalpha * valpha))
		# c = 2/3 * np.sqrt(2 * self.kalpha)
		# return c / (2 * (np.sqrt(valpha) - c * valpha))
		return self.kalpha / (1 + self.kalpha * valpha)

	def vbeta_grad(self, vbeta):
		# return self.kbeta / (4 * self.kbeta * vbeta - 3 * np.sqrt(2 * self.kbeta * vbeta))
		# c = 2/3 * np.sqrt(2 * self.kbeta)
		# return -c / (2 * (np.sqrt(vbeta) - c * vbeta))
		return -self.kbeta / (1 - self.kbeta * vbeta)


	def kalpha_ascent(self):
		return ascent_in_unconstrained_space(self.kalpha_grad, self.kalpha, 0, 1/2, self.learning_rate)
		
	def kbeta_ascent(self):
		return ascent_in_unconstrained_space(self.kbeta_grad, self.kbeta, 0, 1/2, self.learning_rate)
	
	def valpha_ascent(self):
		return ascent_in_unconstrained_space(self.valpha_grad, self.valpha, 0, 1, self.learning_rate)

	def vbeta_ascent(self):
		return ascent_in_unconstrained_space(self.vbeta_grad, self.vbeta, 0, 1, self.learning_rate)


	def kalpha_descent(self):
		return descent_in_unconstrained_space(self.kalpha_grad, self.kalpha, 0, 1/2, self.learning_rate)
		
	def kbeta_descent(self):
		return descent_in_unconstrained_space(self.kbeta_grad, self.kbeta, 0, 1/2, self.learning_rate)
	
	def valpha_descent(self):
		return descent_in_unconstrained_space(self.valpha_grad, self.valpha, 0, 1, self.learning_rate)

	def vbeta_descent(self):
		return descent_in_unconstrained_space(self.vbeta_grad, self.vbeta, 0, 1, self.learning_rate)

	# def kalpha_extra_ascent(self):
	# 	return extra_ascent_in_unconstrained_space(self.kalpha_grad, self.kalpha, 0, 1/2, self.learning_rate)
		
	# def kbeta_extra_ascent(self):
	# 	return extra_ascent_in_unconstrained_space(self.kbeta_grad, self.kbeta, 0, 1/2, self.learning_rate)
	
	# def valpha_extra_ascent(self):
	# 	return extra_ascent_in_unconstrained_space(self.valpha_grad, self.valpha, 0, 1, self.learning_rate)

	# def vbeta_extra_ascent(self):
	# 	return extra_ascent_in_unconstrained_space(self.vbeta_grad, self.vbeta, 0, 1, self.learning_rate)

	# def kalpha_extra_descent(self):
	# 	return extra_descent_in_unconstrained_space(self.kalpha_grad, self.kalpha, 0, 1/2, self.learning_rate)
		
	# def kbeta_extra_descent(self):
	# 	return extra_descent_in_unconstrained_space(self.kbeta_grad, self.kbeta, 0, 1/2, self.learning_rate)
	
	# def valpha_extra_descent(self):
	# 	return extra_descent_in_unconstrained_space(self.valpha_grad, self.valpha, 0, 1, self.learning_rate)

	# def vbeta_extra_descent(self):
	# 	return extra_descent_in_unconstrained_space(self.vbeta_grad, self.vbeta, 0, 1, self.learning_rate)

	def mu(self):
		return self.phi * np.log(1 + 2/3 * np.sqrt(2) * self.kalpha * self.valpha) \
			+ (1 - self.phi) * np.log(1 - 2/3 * np.sqrt(2) * self.kbeta * self.vbeta)

	def inflationary(self):
		return self.mu() > 0

	def quantity(self, state):
		if np.random.random() < self.phi:
			return self.kalpha ** 2 * A(state) 
		return -self.kbeta ** 2 * B(state)

	def delta(self, state, quantity):
		if quantity > 0:
			return state.price * 2/3 * np.sqrt(2) * self.kalpha * self.valpha
		return -state.price * 2/3 * np.sqrt(2) * self.kbeta * self.vbeta

	def generate(inflationary = None):
		# phi = np.random.uniform(0, 1)
		kalpha = np.random.uniform(0, 1/2)
		kbeta = np.random.uniform(0, 1/2)
		valpha = np.random.uniform(0, 1)
		vbeta = np.random.uniform(0, 1)

		s = Strategy(kalpha, kbeta, valpha, vbeta)
		if inflationary is None or s.inflationary() == inflationary: return s
		return Strategy.generate(inflationary)

	def check(self):
		assert 0 < self.phi < 1, f"invalid phi: {self.phi}"
		assert 0 < self.kalpha < 1/2, f"invalid kalpha: {self.kalpha}"
		assert 0 < self.kbeta < 1/2, f"invalid kbeta: {self.kbeta}"
		assert 0 < self.valpha < 1, f"invalid valpha: {self.valpha}"
		assert 0 < self.vbeta < 1, f"invalid vbeta: {self.vbeta}"

	def __str__(self):
		return f"phi={self.phi:.2f} kalpha={self.kalpha:.2f} kbeta={self.kbeta:.2f} valpha={self.valpha:.2f} vbeta={self.vbeta:.2f} (μ={self.mu():.2f})"

# ---- Price evolution

def update(strategy, state, quantity):
	next_price = state.price + strategy.delta(state, quantity)
	return dotdict({
		"price": next_price,
		"taker": dotdict({
			"inv": state.taker.inv + quantity,
			"cash": state.taker.cash - next_price * quantity,
		}),
		"maker": dotdict({
			"inv": state.maker.inv - quantity,
			"cash": state.maker.cash + next_price * quantity,
		}),
	})

# ---- Data visualization

MAX_INV = INITIAL_STATE.taker.inv + INITIAL_STATE.maker.inv
MAX_CASH = INITIAL_STATE.taker.cash + INITIAL_STATE.maker.cash

def plot_all_runs(collaborative_history, competitive_history):
	starting_strategy = collaborative_history[0][0][0]
	
	fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True, sharex=True)
	
	# Set scatter marker size
	mpl.rcParams["lines.markersize"] = 0.5

	plot_run(collaborative_history, axs, "collaborative")
	plot_run(competitive_history, axs, "competitive")

	title = f"""
        Starting strategy {starting_strategy}
        Starting positions [taker I: {INITIAL_STATE.taker.inv}, C: {INITIAL_STATE.taker.cash}] [maker inv: {INITIAL_STATE.maker.inv}, cash: {INITIAL_STATE.maker.cash}]
        Time horizon: {TIME_HORIZON}    Runs: {RUNS}     Learning rate: {starting_strategy.learning_rate:.4f}
        (note: these are log values on linear plots)
    """
	fig.suptitle(title)

	if len(sys.argv) > 1: plt.savefig(sys.argv[1], dpi=200)
	else: plt.show()

def plot_run(history, axs, label):
	# history := [[(strategy, state, quantity) for t in time] for i in run]	
	time = np.arange(TIME_HORIZON)
	
	# The evolution of mu is deterministic, we can just pick the first run
	mu = [strategy.mu() for (strategy, _, _) in history[0]]

	wealth_taker_avg, wealth_taker_std = [], []
	wealth_maker_avg, wealth_maker_std = [], []
	prices_avg, prices_std = [], []
	
	for t in range(TIME_HORIZON):
		across_runs = [history[i][t] for i in range(RUNS)]

		log_wealth_taker = np.log([state.price * state.taker.inv + state.taker.cash \
		    for (_, state, _) in across_runs])
		log_wealth_maker = np.log([state.price * state.maker.inv + state.maker.cash \
		    for (_, state, _) in across_runs])
		log_price = np.log([state.price for (_, state, _) in across_runs])

		wealth_taker_avg.append(np.average(log_wealth_taker))
		wealth_taker_std.append(np.std(log_wealth_taker))

		wealth_maker_avg.append(np.average(log_wealth_maker))
		wealth_maker_std.append(np.std(log_wealth_maker))

		prices_avg.append(np.average(log_price))
		prices_std.append(np.std(log_price))

	wealth_taker_avg = np.array(wealth_taker_avg)
	wealth_taker_std = np.array(wealth_taker_std)

	wealth_maker_avg = np.array(wealth_maker_avg)
	wealth_maker_std = np.array(wealth_maker_std)

	prices_avg = np.array(prices_avg)
	prices_std = np.array(prices_std)

	axs[0, 0].plot(mu, label=label)
	axs[0, 0].set_title("μ")
	axs[0, 0].legend()
	
	axs[0, 1].plot(prices_avg, label=label)
	axs[0, 1].fill_between(time, prices_avg - prices_std, prices_avg + prices_std, alpha=.1)
	axs[0, 1].set_title("Price")
	axs[0, 1].legend()
	
	axs[1, 0].plot(wealth_taker_avg, label=label)
	axs[1, 0].fill_between(time, wealth_taker_avg - wealth_taker_std, wealth_taker_avg + wealth_taker_std, alpha=.1)
	axs[1, 0].set_title("Taker wealth")
	axs[1, 0].legend()

	axs[1, 1].plot(wealth_maker_avg, label=label)
	axs[1, 1].fill_between(time, wealth_maker_avg - wealth_maker_std, wealth_maker_avg + wealth_maker_std, alpha=.1)
	axs[1, 1].set_title("Maker wealth")
	axs[1, 1].legend()

# ---- Play the game

def main():
	# Draw a random strategy
	# starting_strategy = Strategy.generate()

	# Draw a random inflationary strategy
	starting_strategy = Strategy.generate(inflationary=True)
	
	# Draw a random non-inflationary strategy
	# starting_strategy = Strategy.generate(inflationary=True)
	
	# Play this specific strategy
	# starting_strategy = Strategy(
		# kalpha=0.1,
		# kbeta=0.2,
		# valpha=0.4,
		# vbeta=0.2,
	# )

	print(f"""phi={starting_strategy.phi:.2f},
kalpha={starting_strategy.kalpha},
kbeta={starting_strategy.kbeta},
valpha={starting_strategy.valpha},
vbeta={starting_strategy.vbeta},
μ={starting_strategy.mu()}
""")

	runs_collaborative_history = []
	for run in range(RUNS):
		strategy = deepcopy(starting_strategy)
		state = INITIAL_STATE
		collaborative_history = []
		for _ in range(TIME_HORIZON):
			quantity = strategy.quantity(state)
			collaborative_history.append((strategy, state, quantity))
			state = update(strategy, state, quantity)
			sanity_check(state)

			# Update strategy
			strategy = Strategy(
				strategy.kalpha_ascent(),
				strategy.kbeta_ascent(),
				strategy.valpha_ascent(),
				strategy.vbeta_ascent(),
			)
		runs_collaborative_history.append(collaborative_history)

	runs_competitive_history = []
	for run in range(RUNS):
		strategy = deepcopy(starting_strategy)
		state = INITIAL_STATE
		competitive_history = []
		for t in range(TIME_HORIZON):
			quantity = strategy.quantity(state)
			competitive_history.append((strategy, state, quantity))
			state = update(strategy, state, quantity)
			sanity_check(state)

			# Update strategy
			strategy = Strategy(
				strategy.kalpha_ascent(),
				strategy.kbeta_ascent(),
				strategy.valpha_descent(),
				strategy.vbeta_descent(),
			)

			if run == 0 and t % 100 == 0:
				print(strategy)

		runs_competitive_history.append(competitive_history)

	plot_all_runs(runs_collaborative_history, runs_competitive_history)

if __name__ == "__main__":
	main()
