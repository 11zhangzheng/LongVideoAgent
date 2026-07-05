你正在帮助我在 LongVideoAgent 项目中实现一个 training-free 的 Memory-Augmented Verifier-guided Multi-Agent Long Video QA framework。

项目约束：
1. 不修改训练代码。
2. 优先修改 src/evaluation/lvagent/evaluate_api_unified.py 和 evaluate_local_unified.py。
3. 保持 baseline 默认行为不变。所有新功能必须用环境变量开关控制：
   - USE_VIDEO_MEMORY
   - USE_VERIFIER
   - USE_CLIP_REFINER
4. 新增模块放在 src/evaluation/lvagent/：
   - memory.py
   - verifier.py
   - clip_refiner.py
   - metrics_memory.py
5. 每次修改必须是最小 diff，不要大规模重构。
6. 必须保持 TVQA 和 TVQA+ 两种 dataset 都能跑。
7. 所有新增输出必须写入 detailed log，不破坏原有 summary 格式。
8. Memory 只能作为 per-question reasoning state，不能泄漏其他问题答案。
9. Verifier 只能基于已有 subtitle、visual evidence、memory 进行验证，不允许凭常识编造证据。
10. 如果 API 调用失败或 JSON 解析失败，系统不能崩溃，应该 fallback 到原始行为。

实现目标：
第一阶段实现 Search Memory + Clip Memory + Verifier Agent。
第二阶段实现 Verifier-guided clip refinement，包括 expand_previous、expand_next、expand_both、dense_resample_current_clip。
第三阶段再考虑 raw video preprocessing，不要在第一阶段实现。