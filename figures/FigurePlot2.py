import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import scipy.special as sp


# Create sample data
x = np.linspace(0, 10, 100)  # 100 points between 0 and 10
y = 1.5-np.sin(x) # Compute the sine of each x

#Generate tick labels
tickpositions = [i*np.pi/4 for i in range(0,8)]
ticklabels = ['$t_i$']+[r'$t_{' + str(i) + '}$' for i in range(1, 7)]+['$t_f$']

finetickpositions = [np.pi/4 + (np.pi/8)*(1 + root) for root in sp.roots_legendre(6)[0]]
fineticklabels = ['$t_1$']+[r'$t_1^{(' + str(i) + ')}$' for i in range(1, 5)]+['$t_2$']

# Create the main plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label=r'$\Lambda(q,\dot q, t)$', color='b')
ax.set_xlabel('t',fontsize=20)
ax.set_ylabel(r'$\Lambda(q,\dot q, t)$',fontsize=20)
ax.set_xlim(0,2*np.pi)
ax.set_ylim(0,2.6)
ax.set_xticks(tickpositions,ticklabels)
ax.tick_params(axis='both', which='major', labelsize=16)
ax



# Add inset to show a magnified section with higher x-axis tick resolution
ax_inset = inset_axes(ax, width="40%", height="30%",loc='lower right', borderpad=1, bbox_to_anchor=(100,100,600,500))  # Create an inset plot
ax_inset.plot(x, y, color='b')  # Plot the same data on the inset

for i,tick in enumerate(tickpositions):  # Iterate over x-ticks
    if i > 4:
        ax.axvline(x=tick,ymin=0.42, ymax=(1.5-np.sin(tick))/2.6, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=tick, ymin=0, ymax=0.03, color='black', linestyle='--', linewidth=0.5)
    else:
        ax.axvline(x=tick,ymin=0, ymax=(1.5-np.sin(tick))/2.6, color='black', linestyle='--', linewidth=0.5)

# Set a range to magnify on the x-axis and increase the tick resolution
x_start = 7* np.pi/ 32
# Start of zoomed-in section
x_end = 17 * np.pi / 32  # End of zoomed-in section
ax_inset.set_xlim(x_start, x_end)  # Zoom in on the inset x-axis
ax_inset.set_ylim(0, 1)  # Set appropriate y-axis limits

# Increase the resolution of the x-axis ticks in the inset
ax_inset.xaxis.set_ticks(finetickpositions,fineticklabels)  # Apply the locator to the inset's x-axis
ax_inset.yaxis.set_ticks([])

for i,tick in enumerate(finetickpositions):
    ax_inset.axvline(x=tick,ymin=0,ymax=(1.5-np.sin(tick)), color='black', linestyle=':', linewidth=0.5)

# Mark the magnified area on the main plot
mark_inset(ax, ax_inset, loc1=1, loc2=2, fc="none", ec="red")  # Rectangle connecting the inset to the main plot

plt.savefig('cartoon.jpg')
plt.show()  # Display the plot
