import sys
import json
import numpy as np
import matplotlib.pyplot as plt

def usage():
	print("usage: python3 simulation.py (infl | not-infl) [<output-pdf>]")
	exit()

if len(sys.argv) < 2: usage()

# A very small constant
EPS = np.finfo(float).eps

# Set the seed
np.random.seed(11235813)

# Sometimes the price might get too high or too low an Python complains, in
# those chases, either tweak the parameters or reduce the time horizon.
TIME_HORIZON = 1000

# ---- Define feasibility constraints

MIN_INV = 0
MIN_CASH = 0

# ---- Set initial state

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

# ---- Choose a strategy

if sys.argv[1] == "infl":
	PHI = 0.9
	KALPHA = 0.2
	KBETA = 0.2
	VALPHA = 0.4
	VBETA = 0.2
elif sys.argv[1] == "not-infl":
	PHI = 0.7
	KALPHA = 0.2
	KBETA = 0.3
	VALPHA = 0.2
	VBETA = 0.8
	# Case in which mu < 0 but kappa > 0
	# PHI = 0.33987522664445063
	# KALPHA = 0.3532855079045728
	# KBETA = 0.09549689052438949
	# VALPHA = 0.7785194767364301
	# VBETA = 0.6570065381636263
else:
	usage()
	
assert 0 < PHI < 1
assert 0 < KALPHA < 1 / 2
assert 0 < KBETA < 1 / 2
assert 0 < VALPHA < 1
assert 0 < VBETA < 1

KAPPA = PHI * 2 / 3 * np.sqrt(2 * KALPHA * VALPHA) \
	- (1 - PHI) * 2 / 3 * np.sqrt(2 * KBETA * VBETA)

MU = PHI * np.log(1 + 2 / 3 * np.sqrt(2 * KALPHA * VALPHA)) \
	+ (1 - PHI) * np.log(1 - 2 / 3 * np.sqrt(2 * KBETA * VBETA))

def inflationary():
	return MU > 0

# SAFETY_TAKER = MU + (1 - PHI) * np.log(1 - KBETA)
# SAFETY_MAKER = MU + PHI * np.log(1 - KALPHA)

# def safe():
# 	return SAFETY_TAKER > 0 and SAFETY_MAKER > 0

# --- Quantity selection

def A(state):
	return min(state.maker.inv - MIN_INV, (state.taker.cash - MIN_CASH) / state.price)

def B(state):
	return min((state.maker.cash - MIN_CASH) / state.price, state.taker.inv - MIN_INV)

def pick_quantity(state):
	return KALPHA * A(state) if np.random.random() < PHI else -KBETA * B(state)

# ---- Price evolution

def delta(state, quantity):
	if quantity > 0:
		return state.price * 2 / 3 * np.sqrt(2 * KALPHA * VALPHA)
	return -state.price * 2 / 3 * np.sqrt(2 * KBETA * VBETA)

def update(state, quantity):
	next_price = state.price + delta(state, quantity)
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

# def beta_fit(data, min_value, max_value):
	# from scipy.stats import beta
	# """Estimate a beta distribution from the given data"""
	# X = np.linspace(min_value, max_value - min_value, 1000)
	# a, b, loc, scale = beta.fit(data)
	# Y = [beta.pdf(x, a, b, loc, scale) for x in X]
	# return X, Y

def avg(xs):
	return np.cumsum(xs) / np.arange(1, len(xs) + 1)

# ---- Data visualization

MAX_INV = INITIAL_STATE.taker.inv + INITIAL_STATE.maker.inv
MAX_CASH = INITIAL_STATE.taker.cash + INITIAL_STATE.maker.cash

PHI_ALPHA = PHI * KALPHA * (1 + 2 / 3 * np.sqrt(2 * KALPHA * VALPHA))
PHI_BETA = (1 - PHI) * KBETA * (1 - 2 / 3 * np.sqrt(2 * KBETA * VBETA))

PSI = PHI * KALPHA + (1 - PHI) * KBETA

# Only when the strategy is not inflationary
EXPECTED_INVENTORY = MAX_INV * PHI * KALPHA / (PHI * KALPHA + (1 - PHI) * KBETA)

# Only when the strategy is inflationary
EXPECTED_CASH = MAX_CASH * PHI_ALPHA / (PHI_ALPHA + PHI_BETA)

def plot_history(history):
	time = np.arange(TIME_HORIZON)

	As = [A(s) for (s, _) in history]
	Bs = [B(s) for (s, _) in history]
	Ps = [s.price for (s, _) in history]
	Qs = [q for (_, q) in history]
	Ds = [delta(s, q) for (s, q) in history]

	I_Ts = [s.taker.inv for (s, _) in history]
	I_Ms = [s.maker.inv for (s, _) in history]

	C_Ts = [s.taker.cash for (s, _) in history]
	C_Ms = [s.maker.cash for (s, _) in history]

	CP_Ts = [s.taker.cash / s.price for (s, _) in history]
	CP_Ms = [s.maker.cash / s.price for (s, _) in history]

	QI_Ts = [q / s.taker.inv for (s, q) in history]
	QI_Ms = [q / s.maker.inv for (s, q) in history]

	QPs = [q * (s.price + delta(s, q)) for (s, q) in history]

	kappas = [delta(s, q) / s.price for (s, q) in history]

	W_Ts = [s.taker.cash + s.price * s.taker.inv for (s, _) in history]
	W_Ms = [s.maker.cash + s.price * s.maker.inv for (s, _) in history]

	# Expected average price
	EXP_Ps = [INITIAL_STATE.price * (1 + KAPPA) ** t for t in time]

	fig, axs = plt.subplots(
		8, 2,
		figsize=(10, 20),
		constrained_layout=True,
		sharex=True
	)

	axs[0, 0].plot(As)
	axs[0, 0].set_title("A_t")
	axs[0, 0].set_yscale("log")

	axs[0, 1].plot(I_Ms, label="maker's inventory")
	axs[0, 1].plot(CP_Ts, label="taker's cash / price")
	axs[0, 1].set_title("A_t components")
	axs[0, 1].set_yscale("log")
	axs[0, 1].legend()

	axs[1, 0].plot(Bs)
	axs[1, 0].set_title("B_t")
	axs[1, 0].set_yscale("log")

	axs[1, 1].plot(CP_Ms, label="maker's cash / price")
	axs[1, 1].plot(I_Ts, label="taker's inventory")
	axs[1, 1].set_title("B_t components")
	axs[1, 1].set_yscale("log")
	axs[1, 1].legend()

	axs[2, 0].plot(EXP_Ps, alpha=0.5, label="expect", c="purple")
	axs[2, 0].plot(avg(Ps), label="avg", c="purple", linestyle="--")
	axs[2, 0].plot(Ps)
	axs[2, 0].set_title("Price")
	axs[2, 0].set_yscale("log")
	axs[2, 0].legend()

	axs[2, 1].scatter(time, kappas, s=0.5)
	axs[2, 1].plot(avg(kappas), label="avg", c="purple", linestyle="--")
	axs[2, 1].axhline(KAPPA, alpha=0.5, label="expect", c="purple")
	axs[2, 1].axhline(0, c="grey", alpha=0.4)
	axs[2, 1].set_title("kappa")
	axs[2, 1].legend()

	axs[3, 0].scatter(time, Qs, s=0.5)
	axs[3, 0].plot(avg(Qs), label="avg", c="purple", linestyle="--")
	axs[3, 0].axhline(0, c="grey", alpha=0.4)
	axs[3, 0].set_yscale("symlog", linthresh=EPS)
	axs[3, 0].yaxis.get_major_locator().numticks = 6
	axs[3, 0].set_title("Q_t")
	axs[3, 0].legend()

	axs[3, 1].scatter(time, QPs, s=0.5)
	axs[3, 1].plot(avg(QPs), label="avg", c="purple", linestyle="--")
	axs[3, 1].axhline(0, c="grey", alpha=0.4)
	axs[3, 1].set_title("Q_t * P_t+1")
	axs[3, 1].set_yscale("symlog", linthresh=EPS)
	axs[3, 1].yaxis.get_major_locator().numticks = 6
	axs[3, 1].legend()

	axs[4, 0].plot(I_Ts)
	axs[4, 0].plot(avg(I_Ts), label="avg", c="purple", linestyle="--")
	axs[4, 0].axhline(INITIAL_STATE.taker.inv, label="initial", c="C2")
	axs[4, 0].axhline(EXPECTED_INVENTORY, label="expect", c="purple")
	axs[4, 0].set_title("Taker's inventory")
	axs[4, 0].legend()

	axs[4, 1].plot(C_Ts)
	axs[4, 1].plot(avg(C_Ts), label="avg", c="purple", linestyle="--")
	axs[4, 1].axhline((1 - PHI_ALPHA / (PHI_ALPHA + PHI_BETA)) * MAX_CASH, label="expect", c="purple")
	axs[4, 1].axhline(INITIAL_STATE.taker.cash, label="initial", c="C2")
	axs[4, 1].set_title("Taker's cash")
	axs[4, 1].legend()

	axs[5, 0].plot(I_Ms)
	axs[5, 0].plot(avg(I_Ms), label="avg", c="purple", linestyle="--")
	axs[5, 0].axhline(INITIAL_STATE.maker.inv, label="initial", c="C2")
	axs[5, 0].axhline(MAX_INV - EXPECTED_INVENTORY, label="expect", c="purple")
	axs[5, 0].set_title("Maker's inventory")
	axs[5, 0].legend()

	axs[5, 1].plot(C_Ms)
	axs[5, 1].plot(avg(C_Ms), label="avg", c="purple", linestyle="--")
	axs[5, 1].axhline(INITIAL_STATE.maker.cash, label="initial", c="C2")
	axs[5, 1].axhline(PHI_ALPHA / (PHI_ALPHA + PHI_BETA) * MAX_CASH, label="expect", c="purple")
	axs[5, 1].set_title("Maker's cash")
	axs[5, 1].legend()

	axs[6, 0].plot(W_Ms, label="maker")
	axs[6, 0].plot(W_Ts, label="taker")
	axs[6, 0].set_title("Wealth")
	axs[6, 0].set_yscale("log")
	axs[6, 0].legend()

	axs[6, 1].scatter(time, Ds, s=0.5)
	axs[6, 1].plot(avg(Ds), label="avg", c="purple", linestyle="--")
	axs[6, 1].set_yscale("log")
	axs[6, 1].axhline(0, c="grey", alpha=0.4)
	axs[6, 1].set_title("delta_t")
	axs[6, 1].set_yscale("symlog", linthresh=EPS)
	axs[6, 1].yaxis.get_major_locator().numticks = 6
	axs[6, 1].legend()

	axs[7, 0].scatter(time, QI_Ts, label="taker", s=0.5)
	axs[7, 0].scatter(time, QI_Ms, label="maker", s=0.5)
	axs[7, 0].axhline(0, c="grey", alpha=0.4)
	axs[7, 0].set_title("Q_t / I_t")
	axs[7, 0].set_yscale("symlog", linthresh=EPS)
	axs[7, 0].yaxis.get_major_locator().numticks = 6
	axs[7, 0].legend()

	title = f"""
        {PHI=:.2f} {KALPHA=:.2f} {KBETA=:.2f} {VALPHA=:.2f} {VBETA=:.2f} ({"inflationary" if inflationary() else "not inflationary"})
        Starting positions [taker inv {INITIAL_STATE.taker.inv}, cash {INITIAL_STATE.taker.cash}] [maker inv {INITIAL_STATE.maker.inv}, cash {INITIAL_STATE.maker.cash}]
        Starting price {INITIAL_STATE.price}    Time horizon {TIME_HORIZON}
    """
	fig.suptitle(title)

	if len(sys.argv) > 2:
		plt.savefig(sys.argv[2], dpi=200)
	else:
		plt.show()

# ---- Misc

def sanity_check(state):
	assert state.price > 0, f"negative price ({state.price})"
	assert state.taker.inv > MIN_INV, f"taker inv too low ({state.taker.inv})"
	assert state.maker.inv > MIN_INV, f"maker inv too low ({state.maker.inv})"
	assert state.taker.cash > MIN_CASH, f"taker cash too low ({state.taker.cash})"
	assert state.maker.cash > MIN_CASH, f"maker cash too low ({state.maker.cash})"

# ---- Play the game

def main():
	print("μ", MU)
	print("κ", KAPPA)

	if inflationary():
		print("The proposed strategy IS inflationary")
	else:
		print("The proposed strategy IS NOT inflationary")

	state = INITIAL_STATE
	history = []

	for _ in range(TIME_HORIZON):
		quantity = pick_quantity(state)
		history.append((state, quantity))
		state = update(state, quantity)
		sanity_check(state)

	print("Final state")
	print(json.dumps(state, indent=2))

	plot_history(history)

if __name__ == "__main__":
	main()
