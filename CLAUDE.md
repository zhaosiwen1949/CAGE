# CLAUDE.md

CAGE / RoomFormer 点云 → 2D 户型图重建。本文件记录本仓库自定义推理链路的关键约定，供后续开发/调试参考。

## 开发环境

- 本地已配置 conda 环境 **`CAGE`**：开发与调试前先 `conda activate CAGE`，其中已装好 numpy / torch / plyfile / cv2 等依赖。
- 之前该机器无法安装环境、只能做静态分析（`py_compile` + 纯 Python 验证）；**现在可以直接在 `CAGE` 环境里实跑验证**。

## 自定义点云推理链路

入口：`infer_pointcloud.py`（包装脚本 `tools/infer_pointcloud.sh`）。
把任意 `.ply` 点云投影成 256×256 密度图 → RoomFormer 模型 → 房间多边形，输出
`infer_out/{name}_density.png`、`_pred.png`、`_polys.json`。

### 正向变换链（`load_input`）

原始 `.ply` `xyz`
→ `reorder_up_axis(up_axis)`：列置换，把竖直（up）轴移到第 3 列（`y` 轴时 `xyz[:,[0,2,1]]`）
→ 离群点裁剪（见下）
→ `estimate_yaw` + `rotate_floor_plane(+yaw)`：只旋转第 0/1 列，把楼面转正
→ `generate_density`：投影 `xyz[:,:2]` 成密度图（`stru3d_utils.py`，末尾 `density /= density.max()` 归一化到 [0,1]）。

`_polys.json` 里的 `world_mm/world_m` = **重排 + yaw 校正后**的 2D 楼面坐标，不是原始点云坐标。

## 本仓库已做的关键修复 / 增强（infer_pointcloud.py）

针对 MVS / Depth-Anything-3 点云（任意相机系、有飞点、单位常为米）做了以下加强：

1. **Yaw 估计更稳（密度图不再倾斜）**
   - 症状：即便已有 yaw 检测，密度图仍是斜的——根因是 `estimate_yaw` 失效。
   - `estimate_yaw` 现在取**中段高度的墙面点**（`wall_band` 百分位带）而非全体点，
     避开地/顶/家具的面填充淹没薄墙信号；点先中心化、用**固定半径**做投影直方图。
   - `_projection_sharpness` 改用**固定 range `(lo, hi)`** 分箱，消除离群点使分箱变粗、
     以及旋转时包围盒增大带来的偏差。用「归一化直方图平方和」度量投影锐度（曼哈顿世界）。

2. **离群点裁剪不切角（密度图不再缺四个角）**
   - 症状：早期用轴对齐 p2/p98 方盒裁剪，在**旋转前的倾斜系**里切掉了斜放房间的四角 → 八边形。
   - 现方案（旋转不变）：
     - 高度轴：保留 `[pct_low, pct_high]` 百分位带（1D，不影响俯视轮廓，顺带去天花/地噪飞点）。
     - 楼面 (x,y)：**到中位数中心的半径**做 Tukey 栅栏 `rad ≤ Q3 + k·IQR`；半径旋转不变，永不切角，
       且各方向飞点都能抓。飞点去掉后 `generate_density` 的 min/max 自然收紧、画幅贴合。

3. **密度图对比度可调（`--density_gain`）**
   - `generate_density` 用单个最密格子归一化，几个超密格子会把墙压到暗灰。
   - 在量化成 uint8 前 `density = clip(density * gain, 0, 1)`，把峰值 1/gain 以上的格子饱和到白、抬亮墙体。
   - 注意：gain 会**改变模型输入**，可能影响预测多边形；`gain=1.0` 与原图完全等价。

### 相关 CLI 参数（默认值）

- `--up_axis y`：竖直轴（x/y/z）。必须与实际点云一致。
- `--rotation_deg`：手动指定 yaw；给出则跳过自动估计。`--no_align` 关闭自动校正。
- `--align_search_deg 45` / `--align_step_deg 0.5`：yaw 搜索范围与步长。
- `--pct_low 2` / `--pct_high 98`：稳健范围百分位（高度带 + 记录 floor/ceiling）。
- `--crop_iqr_k 3.0`：楼面径向 Tukey 栅栏松紧，越大保留越多（狭长房间端角被削则调大 4~5）。
- `--density_gain 1.0`：密度图对比度增益，量化前乘倍并截断到 [0,1]，建议 2~5。

## 反投影：户型图角点 → 原始 3D（polys_to_3d.py）

`polys_to_3d.py`（包装 `tools/polys_to_3d.sh`）读取 `_polys.json`，把每个 room 的 2D 角点
反投影回**原始点云 3D 坐标系**：撤 yaw → 补 floor/ceiling 双层高度 → 逆列置换，
输出 `_world3d.json` + `_world3d.obj`（墙面环，可与原始 `.ply` 同视图叠加）。

- floor/ceiling 由 `coords_pct_low/high[2]` 恢复：`floor = -coords_pct_high[2]`，`ceiling = -coords_pct_low[2]`（高度轴 yaw 不变），可用 `--floor/--ceiling` 覆盖。
- `--up_axis` 必须与 `infer_pointcloud.py` 当初运行时一致，否则逆置换错误。

## ⚠️ 单位注意

xinghewan 等 MVS 点云是**米**不是毫米。`polys_to_3d.py --units mm` 会差 1000×，按实际单位选 `--units`。
