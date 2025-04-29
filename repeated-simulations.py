import sys
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# A very small constant
TINY = np.finfo(np.float64).tiny
EPS = np.finfo(np.float64).eps

# ---- Proportion of inflationary strategies across runs

INFLATIONARY_PROPORTION = 0.5
DEFAULT_PHI = 0.5

INFL_COLOR = "red"
N_INFL_COLOR = "blue"

# ---- Length and amount of runs

TIME_HORIZON = 500
RUNS = 100

# ---- Feasibility and starting position

MIN_INV = 0
MIN_CASH = 0

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

INITIAL_STATE = dotdict({
	"price": 1,
	"taker": dotdict({"inv": 1, "cash": 5}),
	"maker": dotdict({"inv": 1, "cash": 5}),
})

# ---- State functions

def A(state):
	return min(state.maker.inv - MIN_INV, (state.taker.cash - MIN_CASH) / state.price)

def B(state):
	return min((state.maker.cash - MIN_CASH) / state.price, state.taker.inv - MIN_INV)

 # ---- Strategy

class Strategy:
	def __init__(self, kalpha, kbeta, valpha, vbeta):
		self.phi = DEFAULT_PHI
		self.kalpha = kalpha
		self.kbeta = kbeta
		self.valpha = valpha
		self.vbeta = vbeta
		self.check()

	def check(self):
		assert 0 < self.phi < 1
		assert 0 < self.kalpha < 1/2
		assert 0 < self.kbeta < 1/2
		assert 0 < self.valpha < 1
		assert 0 < self.vbeta < 1

	# def kappa(self):
	# 	return self.phi * 2/3 * np.sqrt(2 * self.kalpha * self.valpha) \
	# 		- (1 - self.phi) * 2/3 * np.sqrt(2 * self.kbeta * self.vbeta)

	def mu(self):
		return self.phi * np.log(1 + 2/3 * np.sqrt(2 * self.kalpha * self.valpha)) \
			+ (1 - self.phi) * np.log(1 - 2/3 * np.sqrt(2 * self.kbeta * self.vbeta))

	def inflationary(self):
		return self.mu() > 0

	def quantity(self, state):
		if np.random.random() < self.phi:
			return self.kalpha * A(state) 
		return -self.kbeta * B(state)

	def delta(self, state, quantity):
		if quantity > 0:
			return state.price * 2/3 * np.sqrt(2 * self.kalpha * self.valpha)
		return -state.price * 2/3 * np.sqrt(2 * self.kbeta * self.vbeta)

	def generate(inflationary):
		# phi = np.random.uniform(EPS, 1)
		kalpha = np.random.uniform(TINY, 1/2)
		kbeta = np.random.uniform(TINY, 1/2)
		valpha = np.random.uniform(TINY, 1)
		vbeta = np.random.uniform(TINY, 1)

		s = Strategy(kalpha, kbeta, valpha, vbeta)
		if s.inflationary() == inflationary: return s
		return Strategy.generate(inflationary)

class StrategyGenerator:
	def __init__(self, inflationary):
		self.inflationary = inflationary

	def next(self):
		self.inflationary -= 1
		return self.reroll()

	def reroll(self):
		return Strategy.generate(self.inflationary >= 0)

	def __len__(self): return int(self.inflationary)

# This is useless, but makes the code easier to change
class StrategyIterator:
	def __init__(self, strategies):
		self.strategies = strategies

	def next(self):
		self.last = self.strategies.pop(0)
		return self.reroll()

	def reroll(self):
		return Strategy(*self.last)
		
	def __len__(self): return len(self.strategies)
		
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

# ---- Data visualization

MAX_INV = INITIAL_STATE.taker.inv + INITIAL_STATE.maker.inv
MAX_CASH = INITIAL_STATE.taker.cash + INITIAL_STATE.maker.cash

# PHI_ALPHA = PHI * KALPHA * (1 + 2 / 3 * np.sqrt(2 * KALPHA * VALPHA))
# PHI_BETA = (1 - PHI) * KBETA * (1 - 2 / 3 * np.sqrt(2 * KBETA * VBETA))

# PSI = PHI * KALPHA + (1 - PHI) * KBETA

# Only when the strategy is inflationary
# EXPECTED_INVENTORY = MAX_INV * PHI * KALPHA / (PHI * KALPHA + (1 - PHI) * KBETA)
# EXPECTED_CASH = MAX_CASH * PHI_ALPHA / (PHI_ALPHA + PHI_BETA)

def plot_all_runs(runs_history):
	fig, axs = plt.subplots(4, 2, figsize=(10, 10), constrained_layout=True, sharex=True)

	min_mu, max_mu = np.log(3/5), np.log(5/3)
	norm = mcolors.Normalize(vmin=min_mu*10, vmax=max_mu*10)

	sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
	cbar = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.03, location="top")
	cbar.set_label("mu", labelpad=-50)

	# Set scatter marker size
	mpl.rcParams["lines.markersize"] = 0.5

	for (strategy, history) in runs_history:
		plot_run(axs, strategy, history)

	for ax in axs.reshape(-1):
		ax.yaxis.get_major_locator().numticks = 6

	freq_infl = sum(strat.inflationary() for (strat, _) in runs_history) / RUNS
	title = f"""
        Starting positions [taker I: {INITIAL_STATE.taker.inv}, C: {INITIAL_STATE.taker.cash}] [maker inv: {INITIAL_STATE.maker.inv}, cash: {INITIAL_STATE.maker.cash}]
        Runs: {RUNS}    Time horizon: {TIME_HORIZON}    Inflationary proportion: {freq_infl:.2f} [expected {INFLATIONARY_PROPORTION}]
    """
	fig.suptitle(title)

	if len(sys.argv) > 1: plt.savefig(sys.argv[1], dpi=200)
	else: plt.show()

def plot_run(axs, strategy, history):
	color = plt.cm.viridis(strategy.mu()*10)
	time = np.arange(TIME_HORIZON)

	P = [s.price for (s, _) in history]

	# Zero-sum reward
	ZT = np.cumsum([1/2 * strategy.delta(s, q) * (s.taker.inv - s.maker.inv) for (s, q) in history])
	ZM = np.cumsum([-1/2 * strategy.delta(s, q) * (s.taker.inv - s.maker.inv) for (s, q) in history])

	# Common reward
	U = np.cumsum([1/2 * strategy.delta(s, q) * MAX_INV for (s, q) in history])

	# Original reward
	RT = np.cumsum([z + u for (z, u) in zip(ZT, U)])
	RM = np.cumsum([z + u for (z, u) in zip(ZM, U)])
	
	axs[0, 0].plot(P, color=color)
	axs[0, 0].set_title("Price")
	axs[0, 0].set_yscale("log")

	axs[0, 1].scatter(time, U, color=color)
	axs[0, 1].set_title("Common reward")
	axs[0, 1].set_yscale("log")

	axs[1, 0].scatter(time, ZT, color=color, marker="+")
	axs[1, 0].set_title("Zero-sum reward taker")
	axs[1, 0].set_yscale("symlog", linthresh=t(ZT))

	axs[1, 1].scatter(time, ZM, color=color, marker="+")
	axs[1, 1].set_title("Zero-sum reward maker")
	axs[1, 1].set_yscale("symlog", linthresh=t(ZT))

	axs[2, 0].scatter(time, RT, color=color)
	axs[2, 0].set_title("Original reward taker")
	axs[2, 0].set_yscale("log")

	axs[2, 1].scatter(time, RM, color=color)
	axs[2, 1].set_title("Original reward maker")
	axs[2, 1].set_yscale("log")

	axs[3, 0].scatter(time, ZT, color=color)
	axs[3, 0].scatter(time, U, color=color, marker="+")
	axs[3, 0].set_title("Common and zero-sum reward taker")
	axs[3, 0].set_yscale("symlog", linthresh=t(ZT))

	axs[3, 1].scatter(time, ZM, color=color)
	axs[3, 1].scatter(time, U, color=color, marker="+")
	axs[3, 1].set_title("Common and zero-sum reward maker")
	axs[3, 1].set_yscale("symlog", linthresh=t(ZT))

# ---- Play the game

def main():
	# print(f"Doing {RUNS} runs...")
	runs_history = []
	strategies = StrategyGenerator(INFLATIONARY_PROPORTION * RUNS)
	# strategies = StrategyIterator([
	#     # ka, kb, va, vb
	# 	[1/2 - EPS, 0 + EPS, 1 - EPS, 0 + EPS], # UU                                   	

	# 	[1/2 - EPS, 0 + EPS, 0 + EPS, 1 - EPS], # ZZ, taker higher inventory
	# 	[1/2 - EPS, 0 + EPS, 1 - EPS, 0 + EPS], # ZU, taker higher inventory
	# 	[1/2 - EPS, 0 + EPS, 0 + EPS, 1 - EPS], # UZ, taker higher inventory

	# 	[0 + EPS,   1/2 - EPS, 1 - EPS, 0 + EPS], # ZZ, maker higher inventory
	# 	[0 + EPS,   1/2 - EPS, 1 - EPS, 0 + EPS], # ZU, maker higher inventory
	# 	[1/2 - EPS, 0 + EPS,   1 - EPS, 0 + EPS], # UZ, maker higher inventory
	# ])

	print(f"Doing {len(strategies)} runs...")
		
	while len(strategies) > 0:
		strategy = strategies.next()

		while True:
			try:
				state = INITIAL_STATE
				history = []
				for _ in range(TIME_HORIZON):
					quantity = strategy.quantity(state)
					history.append((state, quantity))
					state = update(strategy, state, quantity)
					sanity_check(state)
			except AssertionError:
				print("Retrying...")
				strategy = strategies.reroll()
				continue
			break

		runs_history.append((strategy, history))

	plot_all_runs(runs_history)

if __name__ == "__main__":
	main()
