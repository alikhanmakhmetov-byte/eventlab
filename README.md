# EventLab — Probability Project

**Individual project by Makhmetov Alikhan**

EventLab is an English-language Python web app for learning probability through a server-overload example. Its controls, explanations, diagrams, charts, and tables connect each formula to a concrete outcome.

## Run the app

Use Python 3.10 or newer. In a terminal opened in this folder, run:

    py -m pip install -r requirements.txt
    py -m streamlit run app.py

The app opens in a browser at the local address shown in the terminal.

## The model in plain language

One trial represents one hypothetical minute. For each server, the model independently decides whether it overloads. It then counts the overloaded servers. The app repeats this process for the selected number of simulated minutes and compares the observed results with the mathematical probabilities.

The simulation uses generated data. It does not measure real servers, real traffic, or real elapsed time.

| Symbol | Meaning |
| --- | --- |
| n | Total number of servers in the cluster |
| p | Chance that one server overloads in one hypothetical minute |
| k | Exact number of overloaded servers in the outcome being studied |
| m | Number of simulated minutes, or computer-generated trials |
| X | Number of overloaded servers in one trial |

## The four sections

- **Combinatorics** counts how many server groups can produce an outcome, calculates the probability of exactly k overloads, and shows the full distribution.
- **Simulation** repeats the model m times and compares observed frequencies with the theoretical distribution. It reports the mean, median, and standard deviation in plain language.
- **Events** explains conditional probability and shows how independence affects the chance of overloads in consecutive minutes.
- **Formulas** explains the equations and includes a live symbol key using the current settings.

## Main formulas

The number of ways to choose k overloaded servers from n servers is:

    C(n, k) = n! / (k! (n-k)!)

The probability that exactly k servers overload is:

    P(X = k) = C(n, k) p^k (1-p)^(n-k)

The chance that at least one server overloads is:

    P(X >= 1) = 1 - (1-p)^n

The expected number of overloaded servers is n × p. The standard deviation is √(n × p × (1-p)).

## Assumptions

Every server uses the same overload probability p. Server outcomes are independent within a minute, and each simulated minute is independent of the previous one. Real servers can affect one another, so this is a simplified teaching model.

## Technology

Python, Streamlit, NumPy, Pandas, and Plotly. The probability calculations use Python's standard math.comb function and explicit binomial formulas.
