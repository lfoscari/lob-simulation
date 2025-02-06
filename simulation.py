import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# aggiungere oscillazione prezzo
# aggiungere wealth discounted (moltiplicata per gamma^t)
# plotta anche true mean di kappa (una linea)

# --- Set a time horiz:on

TIME_HORIZON = 2_000
AVG_RUNS = 1_000
GAMMA = 0.9

# ---- Pick a metastrategy
# stochastic => buy or sell with prob. phi and 1 - phi
# deterministic => trade the expectation of the above
 
METASTRATEGY = "stochastic"

# ---- Define feasibility constraints

MIN_INV = 0
MIN_CASH = 0

# ---- Set initial state

INITIAL_STATE = {
    "price": 1_000,
    "taker": {
        "inv": 10,
        "cash": 20_000
    },
    "maker": {
        "inv": 10,
        "cash": 20_000
    }
}

# ---- Choose a strategy

PHI = 0.4
KALPHA = 0.4
KBETA = 0.2
VALPHA = 0.6
VBETA = 0.4

assert 0 < PHI < 1
assert 0 < KALPHA < 1/2
assert 0 < KBETA < 1/2
assert 0 < VALPHA < 1
assert 0 < VBETA < 1

KAPPA = PHI * 2/3 * np.sqrt(2 * KALPHA * VALPHA) \
    - (1 - PHI) * 2/3 * np.sqrt(2 * KBETA * VBETA)

def inflationary():
    return PHI * np.log(1 + 2/3 * np.sqrt(2 * KALPHA * VALPHA)) + \
      (1 - PHI) * np.log(1 - 2/3 * np.sqrt(2 * KBETA * VBETA)) \
        > 0

# --- Quantity selection

def A(state):
    return min(
        state["maker"]["inv"] - MIN_INV,
        (state["taker"]["cash"] - MIN_CASH) / state["price"]
    )

def B(state):
    return min(
        (state["maker"]["cash"] - MIN_CASH) / state["price"],
        state["taker"]["inv"] - MIN_INV
    )

def pick_quantity(state):
    match METASTRATEGY:
        case "stochastic":
            # options = [KALPHA * A(state), - KBETA * B(state)]
            # return np.random.choice(options, p = [PHI, 1 - PHI])
            return KALPHA * A(state) if random.random() < PHI else - KBETA * B(state)
        case "deterministic":
            return PHI * KALPHA * A(state) - (1 - PHI) * KBETA * B(state)
        case _:
            raise ValueError("Meta-strategy not found!")

# ---- Price evolution

def delta(state, quantity):
    if quantity > 0:
        return state["price"] * 2/3 * np.sqrt(2 * KALPHA * VALPHA)
    return -state["price"] * 2/3 * np.sqrt(2 * KBETA * VBETA)
    
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
        }
    }

# ---- Data visualization

MAX_INV = INITIAL_STATE["taker"]["inv"] + INITIAL_STATE["maker"]["inv"]
MAX_CASH = INITIAL_STATE["taker"]["cash"] + INITIAL_STATE["maker"]["cash"]

PHI_ALPHA = PHI * KALPHA * (1 + 2/3 * np.sqrt(2 * KALPHA * VALPHA)) 
PHI_BETA = (1 - PHI) * KBETA * (1 - 2/3 * np.sqrt(2 * KBETA * VBETA)) 

def avg(xs): return np.cumsum(xs) / np.arange(1, len(xs) + 1)
   
def plot_history(history, inv, cash):
    As = [A(s) for (s, _) in history]
    Bs = [B(s) for (s, _) in history]
    Ps = [s["price"] for (s, _) in history]
    Qs = [q for (_, q) in history]
    Ds = [delta(s, q) for (s, q) in history]

    QPs = [q * s["price"] for (s, q) in history] 
        
    I_Ts = [s["taker"]["inv"] for (s, _) in history] 
    I_Ms = [s["maker"]["inv"] for (s, _) in history] 

    C_Ts = [s["taker"]["cash"] for (s, _) in history] 
    C_Ms = [s["maker"]["cash"] for (s, _) in history] 
    
    W_Ts = [s["taker"]["cash"] + s["price"] * s["taker"]["inv"] for (s, _) in history]
    W_Ms = [s["maker"]["cash"] + s["price"] * s["maker"]["inv"] for (s, _) in history]

    CP_Ts = [s["taker"]["cash"] / s["price"] for (s, _) in history] 
    CP_Ms = [s["maker"]["cash"] / s["price"] for (s, _) in history] 
    
    # dW_Ts = [(GAMMA ** t) * (s["taker"]["inv"] * s["price"]) for (t, (s, q)) in enumerate(history)]
    # dW_Ms = [(GAMMA ** t) * (s["maker"]["inv"] * s["price"]) for (t, (s, q)) in enumerate(history)]

    kappas = [delta(s, q) / s["price"] for (s, q) in history]

    time = np.arange(TIME_HORIZON)
    fig, axs = plt.subplots(8, 2, figsize=(10, 23), constrained_layout=True) #, sharex=True)
    # fig.tight_layout()
    
    axs[0, 0].plot(time, As)
    axs[0, 0].set_title("A_t")
    axs[0, 0].set_yscale("log")
        
    axs[0, 1].plot(time, I_Ms, label = "maker's inventory")
    axs[0, 1].plot(time, CP_Ts, label = "taker's cash / price")
    axs[0, 1].set_title("A_t components")
    axs[0, 1].set_yscale("log")
    axs[0, 1].legend()

    axs[1, 0].plot(time, Bs)
    axs[1, 0].set_title("B_t")
    axs[1, 0].set_yscale("log")

    axs[1, 1].plot(time, CP_Ms, label = "maker's cash / price")
    axs[1, 1].plot(time, I_Ts, label = "taker's inventory")
    axs[1, 1].set_title("B_t components")
    axs[1, 1].set_yscale("log")
    axs[1, 1].legend()

    axs[2, 0].plot(time, Ps)
    axs[2, 0].set_title("Price")
    axs[2, 0].set_yscale("log")
    
    axs[2, 1].scatter(time, kappas, s = .5)
    axs[2, 1].plot(time, avg(kappas), label = "avg", c = "purple")
    axs[2, 1].axhline(KAPPA, alpha = .5, label = "expect", c = "C3")
    axs[2, 1].axhline(0, c = "grey", alpha = .2)
    axs[2, 1].set_title("kappa")
    axs[2, 1].legend()

    # expected_first_quantity = \
    #     PHI * KALPHA * min(INITIAL_STATE["maker"]["inv"], INITIAL_STATE["taker"]["cash"] / INITIAL_STATE["price"]) \
    #     - (1 - PHI) * KBETA * min(INITIAL_STATE["taker"]["inv"], INITIAL_STATE["maker"]["cash"] / INITIAL_STATE["price"])
    # psi = 1 - PHI * KALPHA - (1 - PHI) * KBETA
    # axs[3, 0].plot(expected_first_quantity * np.cumprod([1] + ([psi] * TIME_HORIZON)), label = "mean", c = "pink")
    min_quantity = max(abs(min(Qs, key=np.abs)), 1e-10)
    axs[3, 0].scatter(time, Qs, s = 0.5)
    axs[3, 0].plot(avg(Qs), label = "avg", c = "purple")
    axs[3, 0].axhline(0, c = "grey", alpha = .2)
    axs[3, 0].set_yscale("symlog", linthresh = min_quantity)
    axs[3, 0].yaxis.get_major_locator().numticks = 6 # fix high num of ticks
    axs[3, 0].set_title("Q_t")
    axs[3, 0].legend()
    
    axs[3, 1].scatter(time, QPs, s = 0.5)
    axs[3, 1].plot(time, avg(QPs), label = "avg", c = "purple")
    axs[3, 1].axhline(0, c = "grey", alpha = .2)
    axs[3, 1].set_title("Q_t * P_t")
    # min_amount = max(min(QPs, key=np.abs), 1e-100)
    # axs[3, 1].set_yscale("symlog", linthresh=min_amount)
    axs[3, 1].yaxis.get_major_locator().numticks = 6 
    axs[3, 1].legend()

    axs[4, 0].plot(time, I_Ts)
    axs[4, 0].axhline(INITIAL_STATE["taker"]["inv"], label = "initial", c = "C2")
    axs[4, 0].set_title("Taker's inventory")
    axs[4, 0].legend()
            
    axs[4, 1].plot(time, C_Ts)
    axs[4, 1].plot(time, avg(C_Ts), label = "avg", c = "purple")
    axs[4, 1].axhline((1 - PHI_ALPHA / (PHI_ALPHA + PHI_BETA)) * MAX_CASH, label = "expect", c = "C3")
    axs[4, 1].set_title("Taker's cash")
    axs[4, 1].legend()
    
    axs[5, 0].plot(time, I_Ms)
    axs[5, 0].axhline(INITIAL_STATE["maker"]["inv"], label = "initial", c = "C2")
    axs[5, 0].set_title("Maker's inventory")
    axs[5, 0].legend()
        
    # cash_seq = [INITIAL_STATE["maker"]["cash"]]
    # for t in time: cash_seq.append(PHI_ALPHA * MAX_CASH + (1 - (PHI_ALPHA + PHI_BETA)) * cash_seq[-1])
    # axs[5, 1].plot(cash_seq, label = "pred", c = "C3")
    axs[5, 1].plot(time, C_Ms)
    axs[5, 1].plot(time, avg(C_Ms), label = "avg", c = "purple")
    axs[5, 1].axhline(PHI_ALPHA / (PHI_ALPHA + PHI_BETA) * MAX_CASH, label = "expect", c = "C3")
    axs[5, 1].set_title("Maker's cash")
    axs[5, 1].legend()

    axs[6, 0].plot(time, W_Ms, label="maker")
    axs[6, 0].plot(time, W_Ts, label="taker")
    axs[6, 0].set_title("Wealth")
    axs[6, 0].set_yscale("log")
    axs[6, 0].legend()
           
    min_quantity = max(min(Ds, key=np.abs), 1e-100)
    axs[6, 1].scatter(time, Ds, s = 0.5)
    axs[6, 1].plot(time, avg(Ds), label = "avg", c = "purple")
    axs[6, 1].set_yscale("log")
    axs[6, 1].set_title("delta_t")
    axs[6, 1].set_yscale("symlog", linthresh = min_quantity)
    axs[6, 1].yaxis.get_major_locator().numticks = 6 # fix high num of ticks
    axs[6, 1].legend()

    [inv_T, inv_M] = zip(*inv)
    axs[7, 0].hist(inv_T, bins=MAX_INV * 2, label = "taker", alpha = 0.7)
    axs[7, 0].hist(inv_M, bins=MAX_INV * 2, label = "maker", alpha = 0.7)
    axs[7, 0].set_title("Distribution of final inventory across runs")
    axs[7, 0].legend()

    [cash_T, cash_M] = zip(*cash)
    axs[7, 1].hist(cash_T, bins=MAX_INV * 2, label = "taker", alpha = 0.7)
    axs[7, 1].hist(cash_M, bins=MAX_INV * 2, label = "maker", alpha = 0.7)
    axs[7, 1].axvline(INITIAL_STATE["taker"]["cash"], label = "initial", c = "purple")
    axs[7, 1].set_title("Distribution of final cash across runs")
    axs[7, 1].legend()

    title = f"""
        {PHI=} {KALPHA=} {KBETA=} {VALPHA=} {VBETA=} ({"inflationary" if inflationary() else "not inflationary"})
        {METASTRATEGY} strategy with time horizon {TIME_HORIZON}
        Starting positions [taker I: {INITIAL_STATE["taker"]["inv"]}, C: {INITIAL_STATE["taker"]["cash"]}] [maker inv: {INITIAL_STATE["maker"]["inv"]}, cash: {INITIAL_STATE["maker"]["cash"]}]
    """
    fig.suptitle(title)

    if len(sys.argv) > 1:
        plt.savefig(sys.argv[1], dpi=200)
    else:
        plt.show()

# ---- Misc

def sanity_check(state):
    assert state["price"] > 0, f"negative price ({state["price"]})"
    assert state["taker"]["inv"] > MIN_INV, f"taker inv too low ({state["taker"]["inv"]})"
    assert state["maker"]["inv"] > MIN_INV, f"maker inv too low ({state["maker"]["inv"]})"
    assert state["taker"]["cash"] > MIN_CASH, f"taker cash too low ({state["taker"]["cash"]})"
    assert state["maker"]["cash"] > MIN_CASH, f"maker cash too low ({state["maker"]["cash"]})"

# ---- Play the game

def main():
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
    print(state)

    # Compute average inventory and cash at the end
    inv, cash = [], []
    for _ in range(AVG_RUNS):
        state = INITIAL_STATE
        for _ in range(TIME_HORIZON):
             quantity = pick_quantity(state)
             state = update(state, quantity)
             sanity_check(state)
        inv.append((state["taker"]["inv"], state["maker"]["inv"]))
        cash.append((state["taker"]["cash"], state["maker"]["cash"]))
         
    plot_history(history, inv, cash)
        
if __name__ == "__main__":
    main()
