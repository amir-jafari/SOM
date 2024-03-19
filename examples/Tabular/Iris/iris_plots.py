from plots import Plots
# Plot the hit histogram
plots = Plots()
fig0, ax0, patch0, text0 = plots.hit_hist( som, df, True)
plt.show()

fig1, ax1, patch1, text1 = plots.plt_top_num()
plt.title('SOM Topology')
plt.show()