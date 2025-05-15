from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

series = signal_data[50:550] 
N = len(series)
t = np.arange(N)
fft_coeffs = np.fft.rfft(series)
freqs = np.fft.rfftfreq(N, d=1)
fft_coeffs /= N
fft_coeffs[0] = 0 
components = []
for f, c in zip(freqs, fft_coeffs):
    comp = np.real(c * np.exp(2j * np.pi * f * t)) * N
    components.append(comp)
components = np.array(components)


fig, ax = plt.subplots(figsize=(16, 9))
line, = ax.plot([], [], lw=2)
ax.set_xlim(200, 300)  
ax.set_ylim(series.min() - 0.5, series.max() + 0.5)

def init():
    line.set_data([], [])
    return (line,)

def update(i):
    y = components[:i+1].sum(axis=0)
    line.set_data(t, y)
    visible_range = len(series)
    start = max(0, N // 2 - visible_range // 2)
    end = min(N, start + visible_range)
    ax.set_xlim(start, end)
    ax.set_ylim(y[start:end].min() - 0.5, y[start:end].max() + 0.5)
    ax.set_title(f"Reconstruction: {i+1}/{len(components)} ")
    return (line,)


frames = range(len(components))
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
ani.save('series_reconstruction_zoom_large.gif', writer=PillowWriter(fps=60), dpi=150) 