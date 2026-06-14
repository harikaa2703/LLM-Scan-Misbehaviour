import re

with open('copy_of_llmscanlevel2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the position before the first 'FULL SCAN'
first_full_scan = content.find('# FULL SCAN')
# Find the line where plot_before_after_intervention ends before that
before_first_full_scan = content[:first_full_scan]
last_plt_show = before_first_full_scan.rfind('# plt.show()')
insert_pos = before_first_full_scan[last_plt_show:].find('\n') + last_plt_show + 1

new_method = '''
    def plot_safe_vs_unsafe_activations(self, comparison):
        if comparison is None:
            print("[INFO] Safe vs unsafe profile not available.")
            return

        if self.safe_activation_profile is None or self.unsafe_activation_profile is None:
            return

        safe_norms = [torch.norm(h.float(), p=2).item()
                      for h in self.safe_activation_profile]
        unsafe_norms = [torch.norm(h.float(), p=2).item()
                        for h in self.unsafe_activation_profile]
        diffs = comparison["per_layer_l2_diff"]
        layers = list(range(len(diffs)))
        most = comparison["most_discriminative_layer"]

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.patch.set_facecolor("#0f0f1a")

        matrix = np.array([safe_norms, unsafe_norms])
        im = axes[0].imshow(matrix, aspect="auto", cmap="coolwarm",
                            interpolation="nearest")
        axes[0].set_yticks([0, 1])
        axes[0].set_yticklabels(["Safe prompt", "Unsafe prompt"],
                                color="white", fontsize=9)
        axes[0].set_title("Hidden State Norms — Safe vs Unsafe Prompt",
                          color="white", fontsize=11)
        axes[0].axvline(most, color="yellow", linestyle="--",
                        alpha=0.8, linewidth=1.5, label=f"Most discriminative L{most}")
        axes[0].tick_params(colors="#aaaaaa")
        plt.colorbar(im, ax=axes[0], label="L2 Norm")

        axes[1].plot(layers, safe_norms, color="#27ae60", linewidth=1.5,
                     label="Safe", alpha=0.9)
        axes[1].plot(layers, unsafe_norms, color="#e74c3c", linewidth=1.5,
                     label="Unsafe", alpha=0.9)
        axes[1].fill_between(layers, safe_norms, unsafe_norms,
                              alpha=0.15, color="#f1c40f")
        axes[1].axvline(most, color="yellow", linestyle="--", alpha=0.6, linewidth=1)
        axes[1].set_ylabel("Hidden State Norm", color="#aaaaaa")
        axes[1].legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        axes[1].set_title("Activation Overlay", color="white", fontsize=11)
        axes[1].set_facecolor("#0f0f1a")
        axes[1].tick_params(colors="#aaaaaa")

        axes[2].fill_between(layers, diffs, alpha=0.4, color="#9b59b6")
        axes[2].plot(layers, diffs, color="#9b59b6", linewidth=1.5)
        axes[2].scatter([most], [diffs[most]], color="yellow", s=100,
                        zorder=5, label=f"Peak L{most}")
        axes[2].set_xlabel("Transformer Layer", color="#aaaaaa")
        axes[2].set_ylabel("L2 Difference", color="#aaaaaa")
        axes[2].set_title("Per-Layer Activation Difference (Safe − Unsafe)",
                          color="white", fontsize=11)
        axes[2].legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        axes[2].set_facecolor("#0f0f1a")
        axes[2].tick_params(colors="#aaaaaa")

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

        plt.suptitle("Safe vs Unsafe Activation Comparison — LLMSCAN",
                     color="white", fontsize=13, y=1.01)
        plt.tight_layout()

'''

content = content[:insert_pos] + new_method + content[insert_pos:]

with open('copy_of_llmscanlevel2.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✓ Added plot_safe_vs_unsafe_activations method')
