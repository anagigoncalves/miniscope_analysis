import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
rectangle = plt.Rectangle((6.5, 0.1), 10, 0.25, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
ax.scatter([1, 2, 3, 4, 5, 6], [0.225, 0.225, 0.225, 0.225, 0.225, 0.225], color='black', s=60)
ax.scatter([7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], color='crimson', s=60)
ax.scatter([7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], color='crimson', s=60)
ax.scatter([17, 18, 19, 20, 21, 22, 23, 24, 25, 26], [0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225], color='royalblue', s=60)
ax.set_xlabel('Trial', fontsize=20)
ax.set_ylabel('Belt speed (m/s)', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Miniscope figures\\for figures\\' + 'split_session_protocol.svg', dpi=128)

# speed_L = np.array([0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225])
# speed_R = np.array([0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225])
# fig, ax = plt.subplots(figsize=(7, 3), tight_layout=True)
# rectangle = plt.Rectangle((6.5, 0.1), 10, 0.25, fc='dimgrey', alpha=0.3)
# plt.gca().add_patch(rectangle)
# for count_t, t in enumerate(trials):
#     ax.scatter(t, speed_L[count_t], color=colors_session[t], s=60)
#     ax.scatter(t, speed_R[count_t], color=colors_session[t], s=60)
# ax.set_xlabel('Trial', fontsize=20)
# ax.set_ylabel('Belt speed (m/s)', fontsize=20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('J:\\Miniscope figures\\for figures\\' + 'split_session_protocol_colors_session', dpi=128)

# fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
# rectangle = plt.Rectangle((3.5, 0.15), 10, 0.25, fc='dimgrey', alpha=0.3)
# plt.gca().add_patch(rectangle)
# ax.scatter([1, 2, 3], [0.275, 0.275, 0.275], color='black', s=60)
# ax.scatter([4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375], color='black', s=60)
# ax.scatter([4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175], color='black', s=60)
# ax.scatter([14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275], color='black', s=60)
# ax.set_xlabel('Trial', fontsize=20)
# ax.set_ylabel('Belt speed (m/s)', fontsize=20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig('J:\\Miniscope figures\\for figures\\' + 'split_session_protocol_stgsn', dpi=128)

fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
ax.scatter([1, 2, 3, 4, 5, 6], [0.225, 0.225, 0.225, 0.225, 0.225, 0.225], color='black', s=60)
ax.scatter([7, 8, 9, 10, 11, 12], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15], color='purple', s=60)
ax.scatter([13, 14, 15, 16, 17, 18], [0.3, 0.3, 0.3, 0.3, 0.3, 0.3], color='orange', s=60)
ax.set_xlabel('Trial', fontsize=20)
ax.set_ylabel('Belt speed (m/s)', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('E:\\Miniscope figures\\for ppt\\' + 'tied_session_protocol', dpi=128)