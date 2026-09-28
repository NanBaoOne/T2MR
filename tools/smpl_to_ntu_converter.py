"""
SMPL 22-joint to NTU-120 25-joint converter for T²MR.
"""

import numpy as np
import os
from tqdm import tqdm


class SMPLToNTUConverter:
    """
    Convert SMPL 22-joint data to the NTU-120 25-joint format used by T²MR.
    """

    def __init__(self):
        """
        SMPL 22关节索引:
        0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1
        4: L_Knee, 5: R_Knee, 6: Spine2, 7: L_Ankle
        8: R_Ankle, 9: Neck, 10: L_Foot, 11: R_Foot
        12: Head, 13: L_Shoulder, 14: R_Shoulder, 15: Head_top
        16: L_Elbow, 17: R_Elbow, 18: L_Wrist, 19: R_Wrist
        20: L_Hand, 21: R_Hand

        NTU-120 25关节索引:
        0: base_spine, 1: mid_spine, 2: neck, 3: head
        4: l_shoulder, 5: l_elbow, 6: l_wrist, 7: l_hand
        8: r_shoulder, 9: r_elbow, 10: r_wrist, 11: r_hand
        12: l_hip, 13: l_knee, 14: l_ankle, 15: l_foot
        16: r_hip, 17: r_knee, 18: r_ankle, 19: r_foot
        20: spine, 21: l_hand_tip, 22: l_thumb, 23: r_hand_tip, 24: r_thumb
        """

        # 直接映射的关节
        self.direct_mapping = {
            # 躯干
            0: 0,  # Pelvis -> base_spine
            3: 20,  # Spine1 -> spine
            6: 1,  # Spine2 -> mid_spine
            9: 2,  # Neck -> neck
            12: 3,  # Head -> head

            # 左臂
            13: 4,  # L_Shoulder -> l_shoulder
            16: 5,  # L_Elbow -> l_elbow
            18: 6,  # L_Wrist -> l_wrist
            20: 7,  # L_Hand -> l_hand

            # 右臂
            14: 8,  # R_Shoulder -> r_shoulder
            17: 9,  # R_Elbow -> r_elbow
            19: 10,  # R_Wrist -> r_wrist
            21: 11,  # R_Hand -> r_hand

            # 左腿
            1: 12,  # L_Hip -> l_hip
            4: 13,  # L_Knee -> l_knee
            7: 14,  # L_Ankle -> l_ankle
            10: 15,  # L_Foot -> l_foot

            # 右腿
            2: 16,  # R_Hip -> r_hip
            5: 17,  # R_Knee -> r_knee
            8: 18,  # R_Ankle -> r_ankle
            11: 19,  # R_Foot -> r_foot
        }

        # 需要插值生成的关节 (NTU索引: 配置)
        self.interpolation_config = {
            # 21: l_hand_tip - 从hand沿着hand-wrist方向延伸
            21: {
                'type': 'extend',
                'base': 7,  # l_hand (NTU索引)
                'direction_from': 6,  # l_wrist
                'direction_to': 7,  # l_hand
                'ratio': 0.3
            },
            # 22: l_thumb - wrist和hand的中点偏移
            22: {
                'type': 'midpoint_offset',
                'joint1': 6,  # l_wrist (NTU索引)
                'joint2': 7,  # l_hand
                'offset_direction_from': 7,  # 从hand
                'offset_direction_to': 6,  # 向wrist
                'ratio': 0.4
            },
            # 23: r_hand_tip
            23: {
                'type': 'extend',
                'base': 11,  # r_hand (NTU索引)
                'direction_from': 10,  # r_wrist
                'direction_to': 11,  # r_hand
                'ratio': 0.3
            },
            # 24: r_thumb
            24: {
                'type': 'midpoint_offset',
                'joint1': 10,  # r_wrist (NTU索引)
                'joint2': 11,  # r_hand
                'offset_direction_from': 11,
                'offset_direction_to': 10,
                'ratio': 0.4
            }
        }

    def convert_single(self, smpl_motion):
        """
        转换单个motion序列

        Args:
            smpl_motion: (T, 22, 3) numpy array

        Returns:
            ntu_motion: (T, 25, 3) numpy array
        """
        T = smpl_motion.shape[0]
        ntu_motion = np.zeros((T, 25, 3), dtype=np.float32)

        # 步骤1: 直接映射已有关节
        for smpl_idx, ntu_idx in self.direct_mapping.items():
            ntu_motion[:, ntu_idx, :] = smpl_motion[:, smpl_idx, :]

        # 步骤2: 插值生成缺失关节
        for ntu_idx, config in self.interpolation_config.items():
            if config['type'] == 'extend':
                # 延伸型: base + direction * ratio
                base_pos = ntu_motion[:, config['base'], :]
                dir_from = ntu_motion[:, config['direction_from'], :]
                dir_to = ntu_motion[:, config['direction_to'], :]
                direction = dir_to - dir_from
                ntu_motion[:, ntu_idx, :] = base_pos + direction * config['ratio']

            elif config['type'] == 'midpoint_offset':
                # 中点偏移型: midpoint + offset * ratio
                joint1_pos = ntu_motion[:, config['joint1'], :]
                joint2_pos = ntu_motion[:, config['joint2'], :]
                midpoint = (joint1_pos + joint2_pos) / 2

                offset_from = ntu_motion[:, config['offset_direction_from'], :]
                offset_to = ntu_motion[:, config['offset_direction_to'], :]
                offset = offset_to - offset_from

                ntu_motion[:, ntu_idx, :] = midpoint + offset * config['ratio']

        return ntu_motion

    def convert_batch(self, input_dir, output_dir, file_list=None):
        """
        批量转换整个数据集

        Args:
            input_dir: SMPL数据目录
            output_dir: NTU输出目录
            file_list: 要转换的文件列表（如果为None则转换所有.npy文件）
        """
        os.makedirs(output_dir, exist_ok=True)

        # 获取文件列表
        if file_list is None:
            file_list = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

        print(f"\n{'=' * 60}")
        print(f"Converting SMPL (22 joints) → NTU-120 (25 joints)")
        print(f"{'=' * 60}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Total files: {len(file_list)}")
        print(f"{'=' * 60}\n")

        success_count = 0
        error_count = 0

        for filename in tqdm(file_list, desc="Converting"):
            try:
                # 读取SMPL数据
                input_path = os.path.join(input_dir, filename)
                smpl_motion = np.load(input_path)

                # 验证格式
                if len(smpl_motion.shape) != 3 or smpl_motion.shape[1] != 22:
                    print(f"⚠ Skipping {filename}: invalid shape {smpl_motion.shape}")
                    error_count += 1
                    continue

                if np.isnan(smpl_motion).any():
                    print(f"⚠ Skipping {filename}: contains NaN")
                    error_count += 1
                    continue

                # 转换
                ntu_motion = self.convert_single(smpl_motion)

                # 保存
                output_path = os.path.join(output_dir, filename)
                np.save(output_path, ntu_motion)

                success_count += 1

            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
                error_count += 1

        # 统计
        print(f"\n{'=' * 60}")
        print(f"Conversion Summary")
        print(f"{'=' * 60}")
        print(f"✓ Successfully converted: {success_count}")
        print(f"✗ Failed: {error_count}")
        print(f"Total: {success_count + error_count}")
        print(f"{'=' * 60}\n")

        return success_count, error_count

    def update_statistics(self, input_dir, output_dir):
        """
        更新Mean和Std统计量（从22关节转换为25关节）

        Args:
            input_dir: 原始统计量目录（包含Mean_raw.npy和Std_raw.npy）
            output_dir: 输出目录
        """
        print(f"\nUpdating statistics files...")

        # 读取原始统计量
        mean_22 = np.load(os.path.join(input_dir, "Mean_raw.npy"))  # (22, 3)
        std_22 = np.load(os.path.join(input_dir, "Std_raw.npy"))  # (22, 3)

        print(f"Original Mean shape: {mean_22.shape}")
        print(f"Original Std shape: {std_22.shape}")

        # 创建25关节的统计量
        mean_25 = np.zeros((25, 3), dtype=np.float32)
        std_25 = np.ones((25, 3), dtype=np.float32)  # 默认std=1

        # 映射已有关节的统计量
        for smpl_idx, ntu_idx in self.direct_mapping.items():
            mean_25[ntu_idx] = mean_22[smpl_idx]
            std_25[ntu_idx] = std_22[smpl_idx]

        # 插值关节使用相邻关节的平均统计量
        for ntu_idx, config in self.interpolation_config.items():
            if config['type'] == 'extend':
                # 使用base关节的统计量
                ref_idx = config['base']
                mean_25[ntu_idx] = mean_25[ref_idx]
                std_25[ntu_idx] = std_25[ref_idx]
            elif config['type'] == 'midpoint_offset':
                # 使用两个关节的平均统计量
                idx1 = config['joint1']
                idx2 = config['joint2']
                mean_25[ntu_idx] = (mean_25[idx1] + mean_25[idx2]) / 2
                std_25[ntu_idx] = (std_25[idx1] + std_25[idx2]) / 2

        # 保存
        np.save(os.path.join(output_dir, "Mean_ntu.npy"), mean_25)
        np.save(os.path.join(output_dir, "Std_ntu.npy"), std_25)

        print(f"✓ Updated Mean shape: {mean_25.shape}")
        print(f"✓ Updated Std shape: {std_25.shape}")
        print(f"✓ Saved to {output_dir}\n")

    def visualize_comparison(self, smpl_motion, ntu_motion, frame_idx=0, save_path=None):
        """
        可视化对比SMPL和NTU骨架

        Args:
            smpl_motion: (T, 22, 3)
            ntu_motion: (T, 25, 3)
            frame_idx: 要可视化的帧
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(18, 8))

        # SMPL骨架连接
        smpl_bones = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (2, 5), (3, 6),
            (4, 7), (5, 8), (6, 9),
            (7, 10), (8, 11), (9, 12),
            (9, 13), (9, 14),
            (13, 16), (14, 17),
            (16, 18), (17, 19),
            (18, 20), (19, 21)
        ]

        # NTU骨架连接
        ntu_bones = [
            (0, 1), (1, 20), (20, 2), (2, 3),  # 脊柱
            (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (6, 22),  # 左臂
            (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (10, 24),  # 右臂
            (0, 12), (12, 13), (13, 14), (14, 15),  # 左腿
            (0, 16), (16, 17), (17, 18), (18, 19)  # 右腿
        ]

        # 绘制SMPL
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_title(f'SMPL Skeleton (22 joints)\nFrame {frame_idx}', fontsize=14, fontweight='bold')

        smpl_frame = smpl_motion[frame_idx]
        x, y, z = smpl_frame[:, 0], smpl_frame[:, 1], smpl_frame[:, 2]

        ax1.scatter(x, z, y, c='red', s=50, alpha=0.8)
        for bone in smpl_bones:
            ax1.plot([x[bone[0]], x[bone[1]]],
                     [z[bone[0]], z[bone[1]]],
                     [y[bone[0]], y[bone[1]]],
                     'b-', linewidth=2, alpha=0.7)

        # 绘制NTU
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set_title(f'NTU-120 Skeleton (25 joints)\nFrame {frame_idx}', fontsize=14, fontweight='bold')

        ntu_frame = ntu_motion[frame_idx]
        x, y, z = ntu_frame[:, 0], ntu_frame[:, 1], ntu_frame[:, 2]

        # 直接映射的关节
        direct_joints = list(self.direct_mapping.values())
        interp_joints = list(self.interpolation_config.keys())

        ax2.scatter(x[direct_joints], z[direct_joints], y[direct_joints],
                    c='green', s=50, alpha=0.8, label='Direct mapping')
        ax2.scatter(x[interp_joints], z[interp_joints], y[interp_joints],
                    c='orange', s=80, marker='^', alpha=1.0, label='Interpolated')

        for bone in ntu_bones:
            ax2.plot([x[bone[0]], x[bone[1]]],
                     [z[bone[0]], z[bone[1]]],
                     [y[bone[0]], y[bone[1]]],
                     'darkgreen', linewidth=2, alpha=0.7)

        ax2.legend()

        # 绘制叠加对比
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.set_title(f'Overlay Comparison\nFrame {frame_idx}', fontsize=14, fontweight='bold')

        # SMPL (蓝色)
        smpl_frame = smpl_motion[frame_idx]
        ax3.scatter(smpl_frame[:, 0], smpl_frame[:, 2], smpl_frame[:, 1],
                    c='blue', s=40, alpha=0.5, label='SMPL (22)')

        # NTU直接映射 (绿色)
        ntu_frame = ntu_motion[frame_idx]
        ax3.scatter(ntu_frame[direct_joints, 0], ntu_frame[direct_joints, 2],
                    ntu_frame[direct_joints, 1],
                    c='green', s=40, alpha=0.5, label='NTU direct (20)')

        # NTU插值 (橙色)
        ax3.scatter(ntu_frame[interp_joints, 0], ntu_frame[interp_joints, 2],
                    ntu_frame[interp_joints, 1],
                    c='orange', s=60, marker='^', alpha=0.8, label='NTU interp (5)')

        ax3.legend()

        # 设置相同的视角
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            ax.set_xlim([-0.8, 0.8])
            ax.set_ylim([-0.8, 0.8])
            ax.set_zlim([-0.5, 1.5])
            ax.view_init(elev=10, azim=-75)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    """
    将 HumanML3D 的 SMPL 22 关节 `new_joints` 批量转换为 NTU 25 关节数据，
    并生成 Mean_ntu.npy / Std_ntu.npy。

    运行前请先执行 `python tools/cal_mean_var.py` 生成 Mean_raw.npy / Std_raw.npy。
    """
    converter = SMPLToNTUConverter()

    data_root = "./data/HumanML3D"
    input_dir = os.path.join(data_root, "new_joints")
    output_dir = os.path.join(data_root, "new_joints_ntu25_v1.0")

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    success, error = converter.convert_batch(input_dir, output_dir)
    converter.update_statistics(input_dir=data_root, output_dir=data_root)

    print(f"Successfully converted: {success}, failed: {error}")


if __name__ == "__main__":
    main()