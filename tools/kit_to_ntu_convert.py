import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


class KITMLToNTUConverter:
    """
    KIT-ML 21关节 到 NTU 25关节的转换

    KIT-ML 21关节定义:
    0: root (pelvis)
    1-2: left/right hip
    3-4: left/right knee
    5-6: left/right ankle
    7: spine1
    8: spine2
    9: spine3
    10: neck
    11: head
    12-13: left/right collar
    14-15: left/right shoulder
    16-17: left/right elbow
    18-19: left/right wrist
    20: head top

    NTU 25关节定义:
    0: spine base (pelvis)
    1: spine mid
    2: neck
    3: head
    4-7: left shoulder, elbow, wrist, hand
    8-11: right shoulder, elbow, wrist, hand
    12-15: left hip, knee, ankle, foot
    16-19: right hip, knee, ankle, foot
    20: spine (lower spine)
    21-24: hand tip and thumb (left/right)
    """

    def __init__(self):
        # KIT-ML到NTU的映射关系
        self.mapping = self._create_mapping()

    def _create_mapping(self):
        """
        创建KIT-ML到NTU的映射关系

        返回字典: {ntu_index: (kit_index, interpolation_method)}
        - 'direct': 直接映射
        - 'mid': 两个关节的中点
        - 'copy': 复制某个关节
        - 'offset': 基于某关节的偏移
        """
        mapping = {
            # 躯干部分
            0: ('direct', 0),  # spine base <- root (pelvis)
            1: ('mid', [7, 8]),  # spine mid <- spine1和spine2中点
            2: ('direct', 10),  # neck <- neck
            3: ('direct', 11),  # head <- head

            # 左臂
            4: ('direct', 14),  # left shoulder <- left shoulder
            5: ('direct', 16),  # left elbow <- left elbow
            6: ('direct', 18),  # left wrist <- left wrist
            7: ('copy', 18),  # left hand <- 复制left wrist (KIT无hand)

            # 右臂
            8: ('direct', 15),  # right shoulder <- right shoulder
            9: ('direct', 17),  # right elbow <- right elbow
            10: ('direct', 19),  # right wrist <- right wrist
            11: ('copy', 19),  # right hand <- 复制right wrist

            # 左腿
            12: ('direct', 1),  # left hip <- left hip
            13: ('direct', 3),  # left knee <- left knee
            14: ('direct', 5),  # left ankle <- left ankle
            15: ('copy', 5),  # left foot <- 复制left ankle

            # 右腿
            16: ('direct', 2),  # right hip <- right hip
            17: ('direct', 4),  # right knee <- right knee
            18: ('direct', 6),  # right ankle <- right ankle
            19: ('copy', 6),  # right foot <- 复制right ankle

            # 额外关节
            20: ('direct', 7),  # spine <- spine1
            21: ('offset', [18, 0.05]),  # left hand tip <- wrist + 偏移
            22: ('offset', [18, 0.03]),  # left thumb <- wrist + 偏移
            23: ('offset', [19, 0.05]),  # right hand tip <- wrist + 偏移
            24: ('offset', [19, 0.03]),  # right thumb <- wrist + 偏移
        }

        return mapping

    def convert_single(self, kit_motion):
        """
        转换单个KIT-ML动作到NTU格式

        Args:
            kit_motion: numpy array, shape (T, 21, 3)

        Returns:
            ntu_motion: numpy array, shape (T, 25, 3)
        """
        T = kit_motion.shape[0]

        # 验证输入格式
        if kit_motion.shape[1] != 21:
            raise ValueError(f"期望21个关节，得到{kit_motion.shape[1]}个")

        # 初始化NTU格式数据
        ntu_motion = np.zeros((T, 25, 3), dtype=np.float32)

        # 遍历每一帧
        for t in range(T):
            frame_kit = kit_motion[t]  # (21, 3)
            frame_ntu = np.zeros((25, 3), dtype=np.float32)

            # 根据映射关系转换
            for ntu_idx, (method, params) in self.mapping.items():
                if method == 'direct':
                    # 直接映射
                    kit_idx = params
                    frame_ntu[ntu_idx] = frame_kit[kit_idx]

                elif method == 'mid':
                    # 中点插值
                    indices = params
                    frame_ntu[ntu_idx] = np.mean([frame_kit[i] for i in indices], axis=0)

                elif method == 'copy':
                    # 复制关节
                    kit_idx = params
                    frame_ntu[ntu_idx] = frame_kit[kit_idx].copy()

                elif method == 'offset':
                    # 偏移关节（用于手指端点）
                    kit_idx, offset_dist = params
                    base_pos = frame_kit[kit_idx]

                    # 计算手腕到肘部的方向向量
                    if kit_idx == 18:  # left wrist
                        elbow_idx = 16
                    else:  # right wrist
                        elbow_idx = 17

                    direction = base_pos - frame_kit[elbow_idx]
                    direction_norm = np.linalg.norm(direction)

                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                        frame_ntu[ntu_idx] = base_pos + direction * offset_dist
                    else:
                        frame_ntu[ntu_idx] = base_pos.copy()

            ntu_motion[t] = frame_ntu

        return ntu_motion

    def convert_file(self, input_path, output_path):
        """
        转换单个文件

        Args:
            input_path: 输入.npy文件路径 (KIT-ML格式)
            output_path: 输出.npy文件路径 (NTU格式)

        Returns:
            bool: 是否成功
        """
        try:
            # 加载数据
            kit_motion = np.load(input_path)

            # 验证格式
            if len(kit_motion.shape) != 3 or kit_motion.shape[1] != 21:
                print(f"⚠️  跳过 {input_path}: 格式不正确 {kit_motion.shape}")
                return False

            # 转换
            ntu_motion = self.convert_single(kit_motion)

            # 保存
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, ntu_motion)

            return True

        except Exception as e:
            print(f"❌ 转换失败 {input_path}: {e}")
            return False

    def convert_batch(self, input_dir, output_dir, file_list=None):
        """
        批量转换目录

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            file_list: 文件列表（可选），如果为None则转换所有.npy文件

        Returns:
            success_count: 成功数量
            error_count: 失败数量
        """
        # 获取文件列表
        if file_list is None:
            input_path = Path(input_dir)
            npy_files = list(input_path.glob("*.npy"))
        else:
            npy_files = [Path(input_dir) / f"{name}.npy" for name in file_list]

        print(f"🔄 开始批量转换...")
        print(f"📁 输入目录: {input_dir}")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 文件数量: {len(npy_files)}\n")

        success_count = 0
        error_count = 0

        for input_path in tqdm(npy_files, desc="转换进度"):
            # 构造输出路径
            relative_path = input_path.relative_to(input_dir)
            output_path = Path(output_dir) / relative_path

            # 转换
            if self.convert_file(str(input_path), str(output_path)):
                success_count += 1
            else:
                error_count += 1

        return success_count, error_count

    def update_statistics(self, input_dir, output_dir):
        """
        重新计算转换后数据的统计量

        Args:
            input_dir: 包含原始Mean.npy和Std.npy的目录
            output_dir: NTU格式数据目录
        """
        print(f"\n📊 重新计算统计量...")

        # 加载所有NTU格式数据
        npy_files = list(Path(output_dir).glob("*.npy"))

        if len(npy_files) == 0:
            print("⚠️  未找到.npy文件")
            return

        all_data = []
        for npy_file in tqdm(npy_files, desc="加载数据"):
            try:
                data = np.load(npy_file)
                if len(data.shape) == 3 and data.shape[1] == 25:
                    all_data.append(data)
            except:
                continue

        if len(all_data) == 0:
            print("⚠️  没有有效数据")
            return

        # 计算统计量
        all_data = np.concatenate(all_data, axis=0)  # (total_frames, 25, 3)

        mean = np.mean(all_data, axis=0)  # (25, 3)
        std = np.std(all_data, axis=0)  # (25, 3)

        # 防止除零
        std = np.where(std < 1e-6, 1.0, std)

        # 保存统计量
        mean_path = os.path.join(input_dir, "Mean_ntu.npy")
        std_path = os.path.join(input_dir, "Std_ntu.npy")

        np.save(mean_path, mean)
        np.save(std_path, std)

        print(f"✓ 保存 Mean.npy: {mean.shape}")
        print(f"✓ 保存 Std.npy: {std.shape}")
        print(f"✓ 统计量基于 {len(all_data)} 帧数据")

    def visualize_mapping(self):
        """
        打印映射关系
        """
        print("\n" + "=" * 70)
        print("KIT-ML (21关节) → NTU (25关节) 映射关系")
        print("=" * 70)

        ntu_names = [
            "spine base", "spine mid", "neck", "head",
            "L shoulder", "L elbow", "L wrist", "L hand",
            "R shoulder", "R elbow", "R wrist", "R hand",
            "L hip", "L knee", "L ankle", "L foot",
            "R hip", "R knee", "R ankle", "R foot",
            "spine", "L hand tip", "L thumb", "R hand tip", "R thumb"
        ]

        kit_names = [
            "root", "L hip", "R hip", "L knee", "R knee",
            "L ankle", "R ankle", "spine1", "spine2", "spine3",
            "neck", "head", "L collar", "R collar",
            "L shoulder", "R shoulder", "L elbow", "R elbow",
            "L wrist", "R wrist", "head top"
        ]

        for ntu_idx in range(25):
            method, params = self.mapping[ntu_idx]
            ntu_name = ntu_names[ntu_idx]

            if method == 'direct':
                kit_name = kit_names[params]
                print(f"{ntu_idx:2d}. {ntu_name:15s} ← {params:2d}. {kit_name} (直接)")
            elif method == 'mid':
                kit_names_list = [kit_names[i] for i in params]
                print(f"{ntu_idx:2d}. {ntu_name:15s} ← 中点({', '.join(kit_names_list)})")
            elif method == 'copy':
                kit_name = kit_names[params]
                print(f"{ntu_idx:2d}. {ntu_name:15s} ← {params:2d}. {kit_name} (复制)")
            elif method == 'offset':
                kit_idx, offset = params
                kit_name = kit_names[kit_idx]
                print(f"{ntu_idx:2d}. {ntu_name:15s} ← {kit_idx:2d}. {kit_name} + 偏移({offset})")

        print("=" * 70 + "\n")


def main():
    """
    将 KIT-ML 的 21 关节 `new_joints` 批量转换为 NTU 25 关节数据，
    并生成 Mean_ntu.npy / Std_ntu.npy。
    """
    converter = KITMLToNTUConverter()

    data_root = "./data/KIT-ML"
    input_dir = os.path.join(data_root, "new_joints")
    output_dir = os.path.join(data_root, "new_joints_ntu25")

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    success, error = converter.convert_batch(input_dir, output_dir)
    converter.update_statistics(input_dir=data_root, output_dir=output_dir)

    print(f"Successfully converted: {success}, failed: {error}")


if __name__ == "__main__":
    main()