import matplotlib
import matplotlib.pyplot as plt

# Update Matplotlib settings to use LaTeX
# matplotlib.rcParams.update({"text.usetex": False, "font.family": "Times new roman"}) # Use latex fonts

# Test with a simple plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title(r'Test Plot: $T_\mathrm{K}$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

# Save the figure
fig.savefig('test_plot1.png', dpi=100)