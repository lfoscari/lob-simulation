import sys
import json
import random
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# A very small constant
EPS = np.finfo(float).eps

# TODO:
# - aggiungere oscillazione prezzo
# - aggiungere wealth discounted (moltiplicata per gamma^t)

TIME_HORIZON = 1_000
AVG_RUNS = 0  # Repeat the simulation more times and plot final inventory and cash

# ---- Define feasibility constraints

MIN_INV = 0
MIN_CASH = 0

# ---- Set initial state

INITIAL_STATE = {
	"price": 1_000,
	"taker": {"inv": 50, "cash": 20_000},
	"maker": {"inv": 50, "cash": 20_000},
}

# ---- Choose a strategy

if sys.argv[1] == "infl":
	PHI = 0.7  # 0.45
	KALPHA = 0.4
	KBETA = 0.3
	VALPHA = 0.6
	VBETA = 0.2
elif sys.argv[1] == "not-infl":
	PHI = 0.2  # 0.4
	KALPHA = 0.4
	KBETA = 0.3
	VALPHA = 0.6
	VBETA = 0.2
else:
	print("The first argument has to be either 'infl' or 'not-infl'")

# Case in which mu < 0 but kappa > 0
# PHI = 0.33987522664445063
# KALPHA = 0.3532855079045728
# KBETA = 0.09549689052438949
# VALPHA = 0.7785194767364301
# VBETA = 0.6570065381636263

assert 0 < PHI < 1
assert 0 < KALPHA < 1 / 2
assert 0 < KBETA < 1 / 2
assert 0 < VALPHA < 1
assert 0 < VBETA < 1

KAPPA = PHI * 2 / 3 * np.sqrt(2 * KALPHA * VALPHA) - (1 - PHI) * 2 / 3 * np.sqrt(2 * KBETA * VBETA)
MU = PHI * np.log(1 + 2 / 3 * np.sqrt(2 * KALPHA * VALPHA)) + (1 - PHI) * np.log(
	1 - 2 / 3 * np.sqrt(2 * KBETA * VBETA)
)


def inflationary():
	return MU > 0


# --- Quantity selection


def A(state):
	return min(
		state["maker"]["inv"] - MIN_INV, (state["taker"]["cash"] - MIN_CASH) / state["price"]
	)


def B(state):
	return min(
		(state["maker"]["cash"] - MIN_CASH) / state["price"], state["taker"]["inv"] - MIN_INV
	)


def pick_quantity(state):
	return KALPHA * A(state) if random.random() < PHI else -KBETA * B(state)


# ---- Price evolution


def delta(state, quantity):
	if quantity > 0:
		return state["price"] * 2 / 3 * np.sqrt(2 * KALPHA * VALPHA)
	return -state["price"] * 2 / 3 * np.sqrt(2 * KBETA * VBETA)


def update(state, quantity):
	next_price = state["price"] + delta(state, quantity)
	return {
		"price": next_price,
		"taker": {
			"inv": state["taker"]["inv"] + quantity,
			"cash": state["taker"]["cash"] - next_price * quantity,
		},
		"maker": {
			"inv": state["maker"]["inv"] - quantity,
			"cash": state["maker"]["cash"] + next_price * quantity,
		},
	}


# ---- Data visualization

MAX_INV = INITIAL_STATE["taker"]["inv"] + INITIAL_STATE["maker"]["inv"]
MAX_CASH = INITIAL_STATE["taker"]["cash"] + INITIAL_STATE["maker"]["cash"]

PHI_ALPHA = PHI * KALPHA * (1 + 2 / 3 * np.sqrt(2 * KALPHA * VALPHA))
PHI_BETA = (1 - PHI) * KBETA * (1 - 2 / 3 * np.sqrt(2 * KBETA * VBETA))

PSI = PHI * KALPHA + (1 - PHI) * KBETA

# Only when the strategy is inflationary
EXPECTED_INVENTORY = MAX_INV * PHI * KALPHA / (PHI * KALPHA + (1 - PHI) * KBETA)
EXPECTED_CASH = MAX_CASH * PHI_ALPHA / (PHI_ALPHA + PHI_BETA)


def beta_fit(data, min_value, max_value):
	"""Estimate a beta distribution from the given data"""
	X = np.linspace(min_value, max_value - min_value, 1000)
	a, b, loc, scale = beta.fit(data)
	Y = [beta.pdf(x, a, b, loc, scale) for x in X]
	return X, Y


def avg(xs):
	return np.cumsum(xs) / np.arange(1, len(xs) + 1)


def plot_history(history, inv, cash, qnt, priceqnt):
	As = [A(s) for (s, _) in history]
	Bs = [B(s) for (s, _) in history]
	Ps = [s["price"] for (s, _) in history]
	Qs = [q for (_, q) in history]
	Ds = [delta(s, q) for (s, q) in history]

	I_Ts = [s["taker"]["inv"] for (s, _) in history]
	I_Ms = [s["maker"]["inv"] for (s, _) in history]

	C_Ts = [s["taker"]["cash"] for (s, _) in history]
	C_Ms = [s["maker"]["cash"] for (s, _) in history]

	CP_Ts = [s["taker"]["cash"] / s["price"] for (s, _) in history]
	CP_Ms = [s["maker"]["cash"] / s["price"] for (s, _) in history]

	QPs = [q * (s["price"] + delta(s, q)) for (s, q) in history]

	kappas = [delta(s, q) / s["price"] for (s, q) in history]

	W_Ts = [s["taker"]["cash"] + s["price"] * s["taker"]["inv"] for (s, _) in history]
	W_Ms = [s["maker"]["cash"] + s["price"] * s["maker"]["inv"] for (s, _) in history]

	# Original reward (T = W^T_{t+1} - W^T_{t})
	DW_Ts = [delta(s, q) * s["taker"]["inv"] for (s, q) in history]
	DW_Ms = [delta(s, q) * s["maker"]["inv"] for (s, q) in history]

	# Zero-sum reward (T = W^T_{t+1} - W^T_{t} - 1/2 delta_t I)
	Z_Ts = [1 / 2 * delta(s, q) * (s["taker"]["inv"] - s["maker"]["inv"]) for (s, q) in history]
	Z_Ms = [1 / 2 * delta(s, q) * (s["maker"]["inv"] - s["taker"]["inv"]) for (s, q) in history]

	# Zero-difference reward (T = M = 1/2(W_{t+1} - W_{t}))
	Us = [1 / 2 * delta(s, q) * MAX_INV for (s, q) in history]

	time = np.arange(TIME_HORIZON)
	fig, axs = plt.subplots(
		9 + (AVG_RUNS > 0), 2, figsize=(10, 23), constrained_layout=True, sharex=(AVG_RUNS <= 0)
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

	axs[2, 0].plot(Ps)
	axs[2, 0].set_title("Price")
	axs[2, 0].set_yscale("log")

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
	axs[3, 0].yaxis.get_major_locator().numticks = 6  # fix high num of ticks
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
	axs[4, 0].axhline(INITIAL_STATE["taker"]["inv"], label="initial", c="C2")
	axs[4, 0].axhline(EXPECTED_INVENTORY, label="expect", c="purple")
	axs[4, 0].set_title("Taker's inventory")
	axs[4, 0].legend()

	axs[4, 1].plot(C_Ts)
	axs[4, 1].plot(avg(C_Ts), label="avg", c="purple", linestyle="--")
	axs[4, 1].axhline(
		(1 - PHI_ALPHA / (PHI_ALPHA + PHI_BETA)) * MAX_CASH, label="expect", c="purple"
	)
	axs[4, 1].axhline(INITIAL_STATE["taker"]["cash"], label="initial", c="C2")
	axs[4, 1].set_title("Taker's cash")
	axs[4, 1].legend()

	axs[5, 0].plot(I_Ms)
	axs[5, 0].plot(avg(I_Ms), label="avg", c="purple", linestyle="--")
	axs[5, 0].axhline(INITIAL_STATE["maker"]["inv"], label="initial", c="C2")
	axs[5, 0].axhline(MAX_INV - EXPECTED_INVENTORY, label="expect", c="purple")
	axs[5, 0].set_title("Maker's inventory")
	axs[5, 0].legend()

	axs[5, 1].plot(C_Ms)
	axs[5, 1].plot(avg(C_Ms), label="avg", c="purple", linestyle="--")
	axs[5, 1].axhline(INITIAL_STATE["maker"]["cash"], label="initial", c="C2")
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
	axs[6, 1].set_title("delta_t")
	axs[6, 1].set_yscale("symlog", linthresh=EPS)
	axs[6, 1].yaxis.get_major_locator().numticks = 6  # fix high num of ticks
	axs[6, 1].legend()

	axs[7, 0].scatter(time, DW_Ts, s=1, label="taker")
	axs[7, 0].scatter(time, DW_Ms, s=1, label="maker")
	axs[7, 0].set_title("Original reward")
	axs[7, 0].set_yscale("symlog", linthresh=EPS)
	axs[7, 0].yaxis.get_major_locator().numticks = 6  # fix high num of ticks
	axs[7, 0].legend()

	axs[7, 1].scatter(time, Z_Ts, s=1, label="taker")
	axs[7, 1].scatter(time, Z_Ms, s=1, label="maker")
	axs[7, 1].axhline(0, c="grey", alpha=0.4)
	axs[7, 1].set_title("Zero-sum reward")
	axs[7, 1].set_yscale("symlog", linthresh=EPS)
	axs[7, 1].yaxis.get_major_locator().numticks = 6  # fix high num of ticks
	axs[7, 1].legend()

	axs[8, 0].scatter(time, Us, s=1)
	axs[8, 0].axhline(0, c="grey", alpha=0.4)
	axs[8, 0].set_title("Common-sum reward")
	axs[8, 0].set_yscale("symlog", linthresh=EPS)
	axs[8, 0].yaxis.get_major_locator().numticks = 6  # fix high num of ticks

	axs[8, 1].scatter(time, Z_Ts, s=1, label="zero-sum taker")
	axs[8, 1].scatter(time, Z_Ms, s=1, label="zero-sum maker")
	axs[8, 1].scatter(time, Us, s=1, label="common-sum")
	axs[8, 1].axhline(0, c="grey", alpha=0.4)
	axs[8, 1].set_title("Zero and common rewards")
	axs[8, 1].set_yscale("symlog", linthresh=EPS)
	axs[8, 1].yaxis.get_major_locator().numticks = 6  # fix high num of ticks
	axs[8, 1].legend()

	if AVG_RUNS > 0:
		# # Beta
		# X, Y = beta_fit(inv_T, EPS, MAX_INV)
		# axs[8, 0].plot(X, Y, label = "beta fit (taker)", alpha = 0.7, color = "C0", linestyle="--")

		# X, Y = beta_fit(inv_M, EPS, MAX_INV)
		# axs[8, 0].plot(X, Y, label = "beta fit (maker)", alpha = 0.7, color = "C1", linestyle="--")
		# # end beta

		[inv_T, inv_M] = zip(*inv)

		axs[-1, 0].hist(inv_T, density=True, bins=20, label="taker", alpha=0.7, color="C0")
		axs[-1, 0].axvline(EXPECTED_INVENTORY, label="taker expect", c="purple")
		axs[-1, 0].axvline(np.mean(inv_T), label="taker avg", c="purple", linestyle="--")

		axs[-1, 0].hist(inv_M, density=True, bins=20, label="maker", alpha=0.7, color="C1")
		axs[-1, 0].axvline(MAX_INV - EXPECTED_INVENTORY, label="maker expect", c="violet")
		axs[-1, 0].axvline(np.mean(inv_M), label="maker avg", c="violet", linestyle="--")

		axs[-1, 0].set_title(f"Distribution of final inventory across {AVG_RUNS} runs")
		axs[-1, 0].legend()

		[cash_T, cash_M] = zip(*cash)

		axs[-1, 1].hist(cash_T, density=True, bins=20, label="taker", alpha=0.7, color="C0")
		axs[-1, 1].axvline(MAX_CASH - EXPECTED_CASH, label="taker expect", c="purple")
		axs[-1, 1].axvline(np.mean(cash_T), label="taker avg", c="purple", linestyle="--")

		axs[-1, 1].hist(cash_M, density=True, bins=20, label="maker", alpha=0.7, color="C1")
		axs[-1, 1].axvline(EXPECTED_CASH, label="maker expect", c="violet")
		axs[-1, 1].axvline(np.mean(cash_M), label="maker avg", c="violet", linestyle="--")

		axs[-1, 1].set_title(f"Distribution of final cash across {AVG_RUNS} runs")
		axs[-1, 1].legend()

		# [pos_qnt, neg_qnt] = zip(*qnt)
		# axs[8, 0].hist(pos_qnt, bins = MAX_INV * 2, label = "pos", alpha = 0.7)
		# axs[8, 0].hist(neg_qnt, bins = MAX_INV * 2, label = "neg", alpha = 0.7)
		# axs[8, 0].axvline(INITIAL_STATE["taker"]["inv"], label = "initial", c = "purple")
		# axs[8, 0].set_title("Signed sum quantities")
		# axs[8, 0].legend()

		# [pos_priceqnt, neg_priceqnt] = zip(*priceqnt)
		# axs[8, 1].hist(pos_priceqnt, bins = MAX_INV * 2, label = "pos", alpha = 0.7)
		# axs[8, 1].hist(neg_priceqnt, bins = MAX_INV * 2, label = "neg", alpha = 0.7)
		# axs[8, 1].axvline(INITIAL_STATE["taker"]["cash"], label = "initial", c = "purple")
		# axs[8, 1].set_title("Signed sum price * quantity")
		# axs[8, 1].legend()

	title = f"""
        {PHI=} {KALPHA=} {KBETA=} {VALPHA=} {VBETA=} ({"inflationary" if inflationary() else "not inflationary"})
        Starting positions [taker I: {INITIAL_STATE["taker"]["inv"]}, C: {INITIAL_STATE["taker"]["cash"]}] [maker inv: {INITIAL_STATE["maker"]["inv"]}, cash: {INITIAL_STATE["maker"]["cash"]}]
        Time horizon {TIME_HORIZON}
    """
	fig.suptitle(title)

	if len(sys.argv) > 2:
		plt.savefig(sys.argv[2], dpi=200)
	else:
		plt.show()


# ---- Misc


def sanity_check(state):
	assert state["price"] > 0, f"negative price ({state['price']})"
	assert state["taker"]["inv"] > MIN_INV, f"taker inv too low ({state['taker']['inv']})"
	assert state["maker"]["inv"] > MIN_INV, f"maker inv too low ({state['maker']['inv']})"
	assert state["taker"]["cash"] > MIN_CASH, f"taker cash too low ({state['taker']['cash']})"
	assert state["maker"]["cash"] > MIN_CASH, f"maker cash too low ({state['maker']['cash']})"


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

	print("Final state after first run")
	print(json.dumps(state, indent=2))

	if AVG_RUNS <= 0:
		plot_history(history, [], [], [], [])
		return

	print(f"Doing {AVG_RUNS} more runs...")

	# Compute average inventory and cash at the end
	inv, cash, qnt, priceqnt = [], [], [], []

	for _ in range(AVG_RUNS):
		state = INITIAL_STATE
		pos_qnt, neg_qnt = 0, 0
		pos_priceqnt, neg_priceqnt = 0, 0

		for _ in range(TIME_HORIZON):
			quantity = pick_quantity(state)
			state = update(state, quantity)

			if quantity > 0:
				pos_qnt += quantity
				pos_priceqnt += state["price"] * quantity
			elif quantity < 0:
				neg_qnt += quantity
				neg_priceqnt += state["price"] * quantity

			sanity_check(state)

		inv.append((state["taker"]["inv"], state["maker"]["inv"]))
		cash.append((state["taker"]["cash"], state["maker"]["cash"]))
		qnt.append((pos_qnt, neg_qnt))
		priceqnt.append((pos_priceqnt, neg_priceqnt))

	plot_history(history, inv, cash, qnt, priceqnt)


if __name__ == "__main__":
	main()
