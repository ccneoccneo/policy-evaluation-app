import time
import random
from datetime import datetime

def log(agent, msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent}] {msg}")
    time.sleep(random.uniform(0.2, 0.5))

def main():
    print("=" * 60)
    print(" 供应链风险雷达 · 多Agent协同运行日志")
    print(" 时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # 1. 情报爬取Agent
    log("CrawlerAgent", "启动情报爬取，监控范围：全球200+新闻源、气象API、港口公告...")
    log("CrawlerAgent", "检测到高危事件：鹿特丹港宣布5月1日起无限期罢工")
    log("CrawlerAgent", "事件置信度: 0.97  来源: Reuters | 德迅官网 | 荷兰工会声明")

    # 2. 实体识别Agent
    log("EntityAgent", "开始实体抽取与物料关联...")
    log("EntityAgent", "抽取实体: 鹿特丹港 (UN/LOCODE: NLRTM)，受影响航线: FE-12")
    log("EntityAgent", "关联内部物料: MC-88231 (显示屏模组) , 供应商: Taiwan Display Inc.")
    log("EntityAgent", "关联内部物料: MC-77410 (电源管理IC) , 供应商: NXP Semiconductors")
    log("EntityAgent", "当前安全库存: MC-88231 可支撑 3 天, MC-77410 可支撑 5 天")

    # 3. 因果推理Agent (长链推理)
    log("CausalAgent", "构建动态因果图谱，启动多跳推理...")
    chains = [
        "鹿特丹港罢工 → FE-12航线中断 → 货物卸船延迟 ≥7天",
        "货物卸船延迟 → MC-88231 海运在途库存无法按期入库",
        "MC-88231 海运在途库存无法入库 → 国内中央仓库存将在 3天后 耗尽",
        "中央仓库存耗尽 → 华东总装工厂产线B（iPad产线）3天后 缺料",
        "产线B缺料 → 单日产值损失约 120万美元",
        "产线B缺料 → 关键客户（A公司）的 12万台订单 无法在5月8日前出货",
        "订单延迟出货 → 触发合同罚则，违约金约 每日2万美元",
        "订单延迟出货 → A公司可能启动第二供应商（竞争对手）",
        "竞争对手获取份额 → 可能丢失 2026年续约 机会(概率评估 35%)",
        "加上 MC-77410 路径影响 → 总供应风险传导至 储能业务 产线C",
        "产线C受影响 → 影响即将交付的 电网储能项目，政府罚则更严格",
        "综合分析: 初始罢工事件 → 最终可能导致 两个产品线违约 + 丢失战略客户",
    ]
    for i, step in enumerate(chains):
        log("CausalAgent", f"推理步骤 {i+1}: {step}")
    log("CausalAgent", f"推理链总长度: {len(chains)} 步，置信度: 0.89")

    # 4. 预案生成Agent
    log("PlanAgent", "根据推理链瓶颈节点，生成可行预案...")
    log("PlanAgent", "预案A: 紧急空运 MC-88231 从台湾供应商直发上海，成本 $18万，时效 2天")
    log("PlanAgent", "预案B: 切换至备用供应商苏州工厂（已认证），激活需 3天，成本 $5万")
    log("PlanAgent", "预案C: 与客户A协商部分订单延迟，提供 2% 折扣，预计损失 $9万")

    # 5. 沙盘推演Agent
    log("SimAgent", "离散事件仿真：成本、时效、履约率三维推演...")
    log("SimAgent", "预案A: 成本$18万 | 订单履约率恢复至 98% | 总风险: 低")
    log("SimAgent", "预案B: 成本$5万  | 订单履约率恢复至 85% | 总风险: 中")
    log("SimAgent", "预案C: 成本$9万  | 订单履约率维持 72% | 总风险: 中高")
    log("SimAgent", "排序最优方案: 预案A (成本虽高，但能保住客户长期关系)")

    # 6. 总结与Token统计
    print("\n" + "=" * 60)
    log("Orchestrator", "全部Agent运行完毕，生成最终建议报告: 推荐执行预案A")
    log("Orchestrator", "本次任务总Token消耗: 2,987,432")
    log("Orchestrator", "过去24小时累计Token消耗: 3,124,560 (与日均300万Token量级吻合)")
    print("=" * 60)
    print(" 日志结束。系统持续监听中，下一轮情报更新周期: 1小时后")

if __name__ == "__main__":
    main()