import json
import re
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score



def has_word(sentence, word):
    pattern = r"\b" + re.escape(word) + r"\b"
    return re.search(pattern, sentence) is not None


class VQAEval:
    def __init__(self):
        self.contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't", "couldnt": "couldn't",
            "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "im": "i'm",
            "ive": "i've", "isnt": "isn't", "itll": "it'll", "lets": "let's",
            "shouldnt": "shouldn't", "thats": "that's", "theyd": "they'd",
            "theyll": "they'll", "theyre": "they're", "theyve": "they've",
            "wasnt": "wasn't", "werent": "weren't", "whats": "what's",
            "wont": "won't", "wouldnt": "wouldn't", "youd": "you'd", "youre": "you're",
            "youve": "you've",
        }

        self.manualMap = {
            "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
            "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
            "nine": "9", "ten": "10"
        }

        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
            "_", "-", ">", "<", "@", "`", ",", "?", "!"
        ]

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) is not None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        return " ".join(outText)

    def evaluate(self, answer, gt_answers):
        answer = self._normalize(answer)
        if isinstance(gt_answers, list):
            gt_answers = [self._normalize(a) for a in gt_answers]
            for gt in gt_answers:
                if has_word(answer, gt):
                    return 1
            return 0
        else:
            gt = self._normalize(gt_answers)
            return 1 if has_word(answer, gt) else 0

    def _normalize(self, text):
        text = text.replace("\n", " ").replace("\t", " ").strip()
        text = self.processPunctuation(text)
        text = self.processDigitArticle(text)
        return text



def evaluate_vqa(pred_path, test_path):
    print(f"Loading predictions from {pred_path}")
    print(f"Loading ground truth from {test_path}")

    # 读取预测文件
    predictions = []
    with open(pred_path, "r") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    # 读取测试集
    with open(test_path, "r") as f:
        gt_data = json.load(f)

    # ground truth 对应
    gt_map = {}
    for entry in gt_data:
        question = entry["input"].strip()
        gt_map[question] = entry["ground_truth_answer"]

    evaluator = VQAEval()
    y_true, y_pred = [], []

    print("Evaluating...")
    for pred in tqdm(predictions):
        q = pred["question"].strip()
        gt = pred["ground_truth_answer"]
        ans = pred["predicted_answer"]

        #test_data_2.json  ground truth
        if q in gt_map:
            gt = gt_map[q]

        label = evaluator.evaluate(ans, gt)
        y_true.append(1)  # ground truth 视为正样本
        y_pred.append(label)  # 预测是否正确

    # 计算指标
    correct = sum(y_pred)
    total = len(y_pred)
    accuracy = correct / total if total > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 输出结果
    print("\n===== Evaluation Results =====")
    print(f"Accuracy: {accuracy:.4f}({correct}/{total})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("==============================")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    pred_file = "/mnt/yunpan/hym/COCO/predictions_new.jsonl"
    test_file = "/mnt/yunpan/hym/COCO/test_data_2.json"
    evaluate_vqa(pred_file, test_file)