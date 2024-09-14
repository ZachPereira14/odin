def save_plot(fig, filename, output_format='png'):
    """Saves the plot to a file."""
    fig.savefig(filename, format=output_format, bbox_inches='tight')
    plt.close(fig)
