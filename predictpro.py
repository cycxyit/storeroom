import json
import os
import math
import time
import re
import requests
from datetime import datetime

# 可选依赖：用于高级模型（LDA/GMM/Parzen）
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm
    from scipy.spatial.distance import cdist
    import numpy as np
except ImportError:
    LinearDiscriminantAnalysis = GaussianMixture = np = None


class SmartPredictor:
    def __init__(self):
        self.reset_data()

    def reset_data(self):
        self.number_history = []
        self.feedback_history = []
        # 初始化所有算法的权重和准确率记录
        self.algorithm_weights = {
            "洛伦兹分类": 1.0,
            "贝叶斯预测": 1.0,
            "马尔可夫链": 1.0,
            "简单趋势": 1.0,
            "全贝叶斯": 1.0,
            "LDA判别": 1.0,
            "GMM混合": 1.0,
            "Parzen窗": 1.0,
            "统计检验": 1.0,
            "K近邻": 1.0,
            "趋势分析": 1.0,  # ← 新增
        }
        self.algorithm_accuracies = {
            "洛伦兹分类": [0, 0],
            "贝叶斯预测": [0, 0],
            "马尔可夫链": [0, 0],
            "简单趋势": [0, 0],
            "全贝叶斯": [0, 0],
            "LDA判别": [0, 0],
            "GMM混合": [0, 0],
            "Parzen窗": [0, 0],
            "统计检验": [0, 0],
            "K近邻": [0, 0],
            "平稳": [0, 0],  # 占位，防止key error if needed, but not used by main predictors
            "趋势分析": [0, 0],
            "综合预测": [0, 0], 
        }

    def load_history(self):
        if os.path.exists("number_history.json"):
            try:
                with open("number_history.json", "r", encoding="utf-8") as f:
                    self.number_history = json.load(f)
            except:
                self.number_history = []

    def save_history(self):
        with open("number_history.json", "w", encoding="utf-8") as f:
            json.dump(self.number_history, f, ensure_ascii=False, indent=2)

    def load_feedback_history(self):
        if os.path.exists("feedback_history.json"):
            try:
                with open("feedback_history.json", "r", encoding="utf-8") as f:
                    self.feedback_history = json.load(f)
            except:
                self.feedback_history = []

    def save_feedback_history(self):
        with open("feedback_history.json", "w", encoding="utf-8") as f:
            json.dump(self.feedback_history, f, ensure_ascii=False, indent=2)

    def restore_weights_from_history(self):
        """试图从最近的反馈记录中恢复模型权重(记忆)"""
        if self.feedback_history:
            last_record = self.feedback_history[-1]
            if "weights" in last_record and isinstance(last_record["weights"], dict):
                # 能够恢复保存的权重
                for k, v in last_record["weights"].items():
                    if k in self.algorithm_weights:
                        self.algorithm_weights[k] = v
            
            if "accuracies" in last_record and isinstance(last_record["accuracies"], dict):
                # 能够恢复保存的准确率统计
                for k, v in last_record["accuracies"].items():
                    if k in self.algorithm_accuracies:
                        self.algorithm_accuracies[k] = v

    def preprocess_data(self, numbers):
        return ["大" if x >= 5 else "小" for x in numbers]

    def lorentzian_classification(self, numbers):
        if len(numbers) < 3:
            return None
        pattern_length = min(3, len(numbers) - 1)
        recent_pattern = tuple(numbers[-pattern_length:])
        next_values = []
        for i in range(len(numbers) - pattern_length):
            if tuple(numbers[i:i + pattern_length]) == recent_pattern:
                if i + pattern_length < len(numbers):
                    next_values.append(numbers[i + pattern_length])
        if not next_values:
            return None
        from collections import Counter
        most_common = Counter(next_values).most_common(1)[0][0]
        return "大" if most_common >= 5 else "小"

    def bayesian_prediction(self, numbers):
        if len(numbers) < 2:
            return None
        last_val = numbers[-1]
        next_big = 0
        next_small = 0
        for i in range(len(numbers) - 1):
            if numbers[i] == last_val:
                if numbers[i + 1] >= 5:
                    next_big += 1
                else:
                    next_small += 1
        if next_big + next_small == 0:
            return None
        return "大" if next_big > next_small else "小"

    def markov_chain_prediction(self, numbers):
        if len(numbers) < 3:
            return None
        order = min(2, len(numbers) - 1)
        while order >= 1:
            state = tuple(numbers[-order:])
            next_vals = []
            for i in range(len(numbers) - order):
                if tuple(numbers[i:i + order]) == state:
                    if i + order < len(numbers):
                        next_vals.append(numbers[i + order])
            if next_vals:
                from collections import Counter
                pred = Counter(next_vals).most_common(1)[0][0]
                return "大" if pred >= 5 else "小"
            order -= 1
        return None

    def simple_trend_analysis(self, numbers):
        if len(numbers) < 3:
            return None
        recent = numbers[-3:]
        big_count = sum(1 for x in recent if x >= 5)
        small_count = len(recent) - big_count
        if big_count > small_count:
            return "大"
        elif small_count > big_count:
            return "小"
        else:
            total_big = sum(1 for x in numbers if x >= 5)
            total_small = len(numbers) - total_big
            return "大" if total_big >= total_small else "小"

    def full_bayes_prediction(self, numbers):
        if len(numbers) < 4:
            return None
        states = self.preprocess_data(numbers)
        for order in range(min(3, len(states) - 1), 0, -1):
            context = tuple(states[-order:])
            count_big, count_small = 0, 0
            total = 0
            for i in range(order, len(states)):
                if tuple(states[i - order:i]) == context:
                    total += 1
                    if i < len(states) - 1:
                        nxt = states[i]
                        if nxt == "大":
                            count_big += 1
                        else:
                            count_small += 1
            if total > 0:
                p_big = (count_big + 1) / (total + 2)
                p_small = (count_small + 1) / (total + 2)
                return "大" if p_big > p_small else "小"
        return None

    def lda_prediction(self, numbers):
        if np is None or len(numbers) < 5:
            return None
        X, y = [], []
        for i in range(3, len(numbers)):
            X.append(numbers[i - 3:i])
            y.append(1 if numbers[i] >= 5 else 0)
        if len(set(y)) < 2:
            return None
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, y)
            last_window = np.array(numbers[-3:]).reshape(1, -1)
            pred = lda.predict(last_window)[0]
            return "大" if pred == 1 else "小"
        except Exception:
            return None

    def gmm_prediction(self, numbers):
        if np is None or len(numbers) < 6:
            return None
        big_windows, small_windows = [], []
        for i in range(3, len(numbers)):
            window = numbers[i - 3:i]
            if numbers[i] >= 5:
                big_windows.append(window)
            else:
                small_windows.append(window)
        if len(big_windows) < 2 or len(small_windows) < 2:
            return None
        try:
            big_gmm = GaussianMixture(n_components=min(2, len(big_windows)), covariance_type='full')
            small_gmm = GaussianMixture(n_components=min(2, len(small_windows)), covariance_type='full')
            big_gmm.fit(big_windows)
            small_gmm.fit(small_windows)
            current = np.array(numbers[-3:]).reshape(1, -1)
            log_big = big_gmm.score_samples(current)[0]
            log_small = small_gmm.score_samples(current)[0]
            return "大" if log_big > log_small else "小"
        except Exception:
            return None

    def parzen_prediction(self, numbers):
        if np is None or len(numbers) < 6:
            return None
        big_windows, small_windows = [], []
        for i in range(3, len(numbers)):
            window = numbers[i - 3:i]
            if numbers[i] >= 5:
                big_windows.append(window)
            else:
                small_windows.append(window)
        if not big_windows or not small_windows:
            return None
        try:
            current = np.array(numbers[-3:])
            def avg_log_density(data, x, h=1.0):
                data = np.array(data)
                dists = cdist([x], data, metric='euclidean')[0]
                dens = np.mean(norm.pdf(dists, scale=h))
                return math.log(dens + 1e-10)
            log_p_big = avg_log_density(big_windows, current)
            log_p_small = avg_log_density(small_windows, current)
            return "大" if log_p_big > log_p_small else "小"
        except Exception:
            return None

    def stat_test_prediction(self, numbers):
        if len(numbers) < 10:
            return None
        recent = numbers[-10:]
        mean_val = sum(recent) / len(recent)
        if mean_val > 5.5:
            return "大"
        elif mean_val < 4.5:
            return "小"
        n_big = sum(1 for x in recent if x >= 5)
        ratio = n_big / len(recent)
        if ratio > 0.65:
            return "大"
        elif ratio < 0.35:
            return "小"
        return None

    def knn_prediction(self, numbers, k_neighbors=5, window_size=3):
        if len(numbers) < window_size + 2:
            return None
        samples = []
        for i in range(window_size, len(numbers)):
            feature = tuple(numbers[i - window_size:i])
            label = "大" if numbers[i] >= 5 else "小"
            samples.append((feature, label))
        if len(samples) < k_neighbors:
            k_neighbors = len(samples)
        if k_neighbors == 0:
            return None
        current_feature = tuple(numbers[-window_size:])
        def euclidean_dist(a, b):
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        distances = []
        for feat, label in samples[:-1]:  # 排除最后一个（无后续）
            d = euclidean_dist(current_feature, feat)
            distances.append((d, label))
        if not distances:
            return None
        distances.sort(key=lambda x: x[0])
        nearest_labels = [label for _, label in distances[:k_neighbors]]
        count_big = sum(1 for lbl in nearest_labels if lbl == "大")
        return "大" if count_big > len(nearest_labels) / 2 else "小"

    def trend_analysis_prediction(self, numbers):
        """
        基于历史序列的趋势预测：
        - 计算最近 N 个点的线性斜率
        - 判断整体趋势方向
        - 预测下一个值是"上升"、"下降"还是"平稳"
        - 并映射为对"大/小"的间接判断
        """
        if len(numbers) < 3:
            return None
        
        # 使用最近5个点（或全部）
        window = numbers[-min(5, len(numbers)):]
        n = len(window)
        
        # 计算一阶差分（变化量）
        diffs = [window[i] - window[i-1] for i in range(1, n)]
        avg_diff = sum(diffs) / len(diffs) if diffs else 0
        
        # 判断趋势
        if avg_diff > 0.8:
            trend = "上升"
        elif avg_diff < -0.8:
            trend = "下降"
        else:
            trend = "平稳"
        
        # 基于当前最后一个值 + 趋势 → 预测下一个是"大"还是"小"
        last_val = window[-1]
        if trend == "上升":
            predicted_next = min(9, last_val + max(1, round(avg_diff)))
        elif trend == "下降":
            predicted_next = max(0, last_val + min(-1, round(avg_diff)))
        else:
            predicted_next = last_val
        
        return "大" if predicted_next >= 5 else "小"

    def ensemble_prediction(self, numbers):
        predictors = [
            ("洛伦兹分类", self.lorentzian_classification),
            ("贝叶斯预测", self.bayesian_prediction),
            ("马尔可夫链", self.markov_chain_prediction),
            ("简单趋势", self.simple_trend_analysis),
            ("全贝叶斯", self.full_bayes_prediction),
            ("LDA判别", self.lda_prediction),
            ("GMM混合", self.gmm_prediction),
            ("Parzen窗", self.parzen_prediction),
            ("统计检验", self.stat_test_prediction),
            ("K近邻", self.knn_prediction),
            ("趋势分析", self.trend_analysis_prediction),  # ← 新增这一行
        ]
        votes = {}
        detail_items = []
        for name, func in predictors:
            weight = self.algorithm_weights.get(name, 1.0)
            try:
                pred = func(numbers)
                if pred in ["大", "小"]:
                    votes[pred] = votes.get(pred, 0) + weight
                    msg = f"{name}({weight:.2f}): {pred}"
                else:
                    msg = f"{name}: N/A"
            except Exception:
                msg = f"{name}: Error"
            detail_items.append((weight, msg))
        
        # 按权重降序排列
        detail_items.sort(key=lambda x: x[0], reverse=True)
        details = [x[1] for x in detail_items]
        print("各投票者意见: \n" + " |\n ".join(details))
        if not votes:
            return "小"
        return max(votes, key=votes.get)

    def update_algorithm_weights(self, actual_result, previous_predictions):
        for name, pred in previous_predictions.items():
            if name in self.algorithm_accuracies:
                correct, total = self.algorithm_accuracies[name]
                total += 1
                if pred == actual_result:
                    correct += 1
                self.algorithm_accuracies[name] = [correct, total]
                accuracy = correct / total if total > 0 else 0.5
                self.algorithm_weights[name] = 0.8 + 0.4 * accuracy

    def record_feedback(self, prediction, actual, weights_snapshot, accuracies_snapshot):
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "actual": actual,
            "correct": prediction == actual,
            "weights": weights_snapshot,
            "accuracies": accuracies_snapshot
        }
        self.feedback_history.append(feedback)

    def get_detailed_analysis(self):
        if not self.number_history:
            return "暂无数据。"
        total = len(self.number_history)
        big_count = sum(1 for x in self.number_history if x >= 5)
        small_count = total - big_count
        from collections import Counter
        freq = Counter(self.number_history)
        analysis = f"总次数: {total}\n"
        analysis += f"大 ({big_count}): {big_count/total:.1%} | 小 ({small_count}): {small_count/total:.1%}\n"
        analysis += "数值分布:\n"
        for num in range(10):
            count = freq.get(num, 0)
            bar = "█" * (count // max(1, total//50))
            analysis += f"{num:2d}: {count:3d} {bar}\n"
        analysis += "\n算法性能（正确/总数）:\n"
        for name, (corr, tot) in self.algorithm_accuracies.items():
            if tot > 0:
                acc = corr / tot
                analysis += f"{name}: {corr}/{tot} ({acc:.1%})\n"
        states = self.preprocess_data(self.number_history)
        max_streak = 1
        current = 1
        for i in range(1, len(states)):
            if states[i] == states[i-1]:
                current += 1
            else:
                max_streak = max(max_streak, current)
                current = 1
        max_streak = max(max_streak, current)
        analysis += f"\n最长连续相同结果: {max_streak}"
        return analysis

    def get_feedback_stats(self):
        if not self.feedback_history:
            return "暂无反馈记录。"
        total = len(self.feedback_history)
        correct = sum(1 for f in self.feedback_history if f["correct"])
        return f"基于用户反馈的准确率: {correct}/{total} ({correct/total:.1%})"

    def get_stats_summary(self, predictions=None):
        """获取所有算法的准确率统计摘要
        
        Args:
            predictions: 可选字典，包含各算法最后一次的预测结果
        """
        stats_lines = []
        stats_lines.append("\n" + "=" * 70)
        stats_lines.append("📊 算法准确率统计（按权重从高到低排序）")
        stats_lines.append("=" * 70)
        
        # 单个算法的准确率和权重 - 按权重从大到小排序
        stats_lines.append("【各算法性能】")
        algo_list = []
        for name, (corr, tot) in self.algorithm_accuracies.items():
            # 排除"综合预测"，单独显示
            if name != "综合预测":
                weight = self.algorithm_weights.get(name, 1.0)
                algo_list.append((name, corr, tot, weight))
        
        # 按权重从大到小排序
        algo_list.sort(key=lambda x: x[3], reverse=True)
        
        for name, corr, tot, weight in algo_list:
            pred_result = predictions.get(name, "N/A") if predictions else "N/A"
            if tot > 0:
                acc = corr / tot
                stats_lines.append(f"权重: {weight:6.2f} | {name:12s}: {corr:3d}/{tot:3d} ({acc:6.1%}) | 预测: {pred_result}")
            else:
                stats_lines.append(f"权重: {weight:6.2f} | {name:12s}: 0/0 (N/A) | 预测: {pred_result}")
        
        # 综合预测的准确率
        stats_lines.append("\n【综合预测性能】")
        if self.feedback_history:
            ensemble_correct = sum(1 for f in self.feedback_history if f["correct"])
            ensemble_total = len(self.feedback_history)
            ensemble_acc = ensemble_correct / ensemble_total if ensemble_total > 0 else 0
            ensemble_pred = predictions.get("综合预测", "N/A") if predictions else "N/A"
            stats_lines.append(f"{'综合预测':12s}: {ensemble_correct:3d}/{ensemble_total:3d} ({ensemble_acc:6.1%}) | 预测: {ensemble_pred}")
        else:
            ensemble_pred = predictions.get("综合预测", "N/A") if predictions else "N/A"
            stats_lines.append(f"{'综合预测':12s}: 0/0 (N/A) | 预测: {ensemble_pred}")
        
        stats_lines.append("=" * 70)
        return "\n".join(stats_lines)

    def fetch_latest_number(self, url="https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json", timeout=10):
        """从API爬取最新的 number 值"""
        try:
            # 设置请求头，模拟浏览器请求以避免403错误
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://draw.ar-lottery01.com/',
                'Origin': 'https://draw.ar-lottery01.com',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin'
            }
            response = requests.get(url, headers=headers, timeout=timeout, verify=True)
            response.raise_for_status()
            text = response.text
            # 使用正则表达式提取第一个（最新的）number 值
            match = re.search(r'"number"\s*:\s*"(\d+)"', text)
            if match:
                return int(match.group(1))
        except requests.exceptions.RequestException as e:
            print(f"爬虫错误: {e}")
        except Exception as e:
            print(f"解析错误: {e}")
        return None


def main():
    predictor = SmartPredictor()
    predictor.load_history()
    predictor.load_feedback_history()
    predictor.restore_weights_from_history()
    print("=" * 50)
    print("🔢 智能数字预测系统 v3.1（集成爬虫）")
    print("每30秒从API爬取最新数据并自动预测")
    print("按 Ctrl+C 停止程序")
    print("=" * 50)

    url = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
    last_number = None

    try:
        while True:
            # 从API爬取最新的 number
            current_number = predictor.fetch_latest_number(url)
            
            if current_number is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 获取数据失败，等待重试...")
                time.sleep(30)
                continue
            
            # 如果是新数据才进行预测
            if current_number != last_number:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📥 获取新数据: {current_number}")
                actual_result = "大" if current_number >= 5 else "小"
                
                # 如果历史记录足够，进行预测
                if len(predictor.number_history) > 1:
                    print(f"📊 当前历史序列: {predictor.number_history[-10:]}")  # 显示最近10个
                    prediction = predictor.ensemble_prediction(predictor.number_history)
                    print(f"🔮 预测本次输入 ({current_number}) 预测: 「{prediction}」")
                    
                    # 单独计算趋势方向用于展示
                    trend_pred = predictor.trend_analysis_prediction(predictor.number_history)
                    if trend_pred is not None:
                        window = predictor.number_history[-min(5, len(predictor.number_history)):]
                        diffs = [window[i] - window[i-1] for i in range(1, len(window))]
                        avg_diff = sum(diffs) / len(diffs) if diffs else 0
                        if avg_diff > 0.8:
                            trend_dir = "📈 上升"
                        elif avg_diff < -0.8:
                            trend_dir = "📉 下降"
                        else:
                            trend_dir = "➡️ 平稳"
                        print(f"🧭 趋势分析: {trend_dir} (均变: {avg_diff:+.2f})")
                    
                    # 获取各模型对本次输入的预测（用于反馈）
                    previous_predictions = {}
                    predictors = [
                        ("洛伦兹分类", predictor.lorentzian_classification),
                        ("贝叶斯预测", predictor.bayesian_prediction),
                        ("马尔可夫链", predictor.markov_chain_prediction),
                        ("简单趋势", predictor.simple_trend_analysis),
                        ("全贝叶斯", predictor.full_bayes_prediction),
                        ("LDA判别", predictor.lda_prediction),
                        ("GMM混合", predictor.gmm_prediction),
                        ("Parzen窗", predictor.parzen_prediction),
                        ("统计检验", predictor.stat_test_prediction),
                        ("K近邻", predictor.knn_prediction),
                        ("趋势分析", predictor.trend_analysis_prediction),
                    ]
                    for name, func in predictors:
                        try:
                            pred = func(predictor.number_history)
                            if pred in ["大", "小"]:
                                previous_predictions[name] = pred
                        except:
                            pass
                    
                    # 更新模型权重并记录反馈
                    previous_predictions["综合预测"] = prediction
                    predictor.update_algorithm_weights(actual_result, previous_predictions)
                    weights_snap = predictor.algorithm_weights.copy()
                    acc_snap = {k: v.copy() for k, v in predictor.algorithm_accuracies.items()}
                    predictor.record_feedback(prediction, actual_result, weights_snap, acc_snap)
                    predictor.save_feedback_history()
                    
                    # 显示准确率统计
                    print(f"✅ 实际结果: {actual_result} | 预测是否正确: {'✓' if prediction == actual_result else '✗'}")
                    
                    # 显示全面的统计摘要（传入各算法的预测结果）
                    print(predictor.get_stats_summary(previous_predictions))
                    
                    # 最后一行：最终预测结果
                    print(f"\n🎯 最终预测结果: 【{prediction}】\n")
                else:
                    print("📌 已接收首个数据，等待更多输入以开始预测...")
                
                # 将新数字加入历史
                predictor.number_history.append(current_number)
                predictor.save_history()
                last_number = current_number
            
            # 等待30秒后继续
            time.sleep(30)
    
    except KeyboardInterrupt:
        print("\n\n停止程序...")
        predictor.save_history()
        predictor.save_feedback_history()
        print("数据已保存，再见！")


if __name__ == "__main__":
    main()
