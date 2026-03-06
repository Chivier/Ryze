"""Metrics Calculator for Model Evaluation"""
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import string
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various evaluation metrics"""

    @staticmethod
    def calculate_bleu(reference: str, hypothesis: str, n_gram: int = 4) -> float:
        """Calculate BLEU score"""
        def get_ngrams(text: str, n: int) -> List[tuple]:
            tokens = text.lower().split()
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if len(hyp_tokens) == 0:
            return 0.0

        scores = []
        for n in range(1, min(n_gram + 1, len(hyp_tokens) + 1)):
            ref_ngrams = Counter(get_ngrams(reference, n))
            hyp_ngrams = Counter(get_ngrams(hypothesis, n))

            matches = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())

            if total == 0:
                score = 0
            else:
                score = matches / total

            scores.append(score)

        if not scores:
            return 0.0

        # Calculate brevity penalty
        bp = 1.0
        if len(hyp_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / len(hyp_tokens))

        # Geometric mean of n-gram precisions
        bleu = bp * np.exp(np.mean([np.log(s) if s > 0 else float('-inf') for s in scores]))

        return bleu if not np.isnan(bleu) and bleu != float('-inf') else 0.0

    @staticmethod
    def calculate_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores"""
        def lcs_length(x: List[str], y: List[str]) -> int:
            m, n = len(x), len(y)
            lcs = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        lcs[i][j] = lcs[i-1][j-1] + 1
                    else:
                        lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])

            return lcs[m][n]

        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not hyp_tokens or not ref_tokens:
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}

        # ROUGE-1 (unigram)
        ref_unigrams = Counter(ref_tokens)
        hyp_unigrams = Counter(hyp_tokens)
        unigram_overlap = sum((ref_unigrams & hyp_unigrams).values())

        rouge_1_p = unigram_overlap / len(hyp_tokens) if hyp_tokens else 0
        rouge_1_r = unigram_overlap / len(ref_tokens) if ref_tokens else 0
        rouge_1_f = 2 * rouge_1_p * rouge_1_r / (rouge_1_p + rouge_1_r) if (rouge_1_p + rouge_1_r) > 0 else 0

        # ROUGE-2 (bigram)
        ref_bigrams = Counter(zip(ref_tokens[:-1], ref_tokens[1:]))
        hyp_bigrams = Counter(zip(hyp_tokens[:-1], hyp_tokens[1:]))
        bigram_overlap = sum((ref_bigrams & hyp_bigrams).values())

        rouge_2_p = bigram_overlap / len(list(hyp_bigrams)) if hyp_bigrams else 0
        rouge_2_r = bigram_overlap / len(list(ref_bigrams)) if ref_bigrams else 0
        rouge_2_f = 2 * rouge_2_p * rouge_2_r / (rouge_2_p + rouge_2_r) if (rouge_2_p + rouge_2_r) > 0 else 0

        # ROUGE-L (LCS)
        lcs_len = lcs_length(ref_tokens, hyp_tokens)
        rouge_l_p = lcs_len / len(hyp_tokens) if hyp_tokens else 0
        rouge_l_r = lcs_len / len(ref_tokens) if ref_tokens else 0
        rouge_l_f = 2 * rouge_l_p * rouge_l_r / (rouge_l_p + rouge_l_r) if (rouge_l_p + rouge_l_r) > 0 else 0

        return {
            'rouge-1': rouge_1_f,
            'rouge-2': rouge_2_f,
            'rouge-l': rouge_l_f
        }

    @staticmethod
    def calculate_exact_match(reference: str, hypothesis: str) -> float:
        """Calculate exact match score"""
        def normalize_text(text: str) -> str:
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text

        return float(normalize_text(reference) == normalize_text(hypothesis))

    @staticmethod
    def calculate_perplexity(log_likelihoods: List[float], num_tokens: int) -> float:
        """Calculate perplexity from log likelihoods"""
        if num_tokens == 0:
            return float('inf')

        avg_log_likelihood = sum(log_likelihoods) / num_tokens
        perplexity = np.exp(-avg_log_likelihood)

        return perplexity

    @staticmethod
    def calculate_all_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """Calculate all metrics for a batch of examples"""
        if len(references) != len(hypotheses):
            raise ValueError("Number of references and hypotheses must match")

        all_metrics = {
            'bleu': [],
            'rouge-1': [],
            'rouge-2': [],
            'rouge-l': [],
            'exact_match': []
        }

        for ref, hyp in zip(references, hypotheses):
            # BLEU
            bleu = MetricsCalculator.calculate_bleu(ref, hyp)
            all_metrics['bleu'].append(bleu)

            # ROUGE
            rouge_scores = MetricsCalculator.calculate_rouge(ref, hyp)
            all_metrics['rouge-1'].append(rouge_scores['rouge-1'])
            all_metrics['rouge-2'].append(rouge_scores['rouge-2'])
            all_metrics['rouge-l'].append(rouge_scores['rouge-l'])

            # Exact match
            em = MetricsCalculator.calculate_exact_match(ref, hyp)
            all_metrics['exact_match'].append(em)

        # Calculate averages
        avg_metrics = {
            metric: np.mean(scores) for metric, scores in all_metrics.items()
        }

        # Add standard deviations
        std_metrics = {
            f"{metric}_std": np.std(scores) for metric, scores in all_metrics.items()
        }

        return {**avg_metrics, **std_metrics}

    @staticmethod
    def calculate_diversity_metrics(texts: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics for generated texts"""
        all_tokens = []
        all_bigrams = []
        all_trigrams = []

        for text in texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)

            if len(tokens) >= 2:
                bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
                all_bigrams.extend(bigrams)

            if len(tokens) >= 3:
                trigrams = [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens)-2)]
                all_trigrams.extend(trigrams)

        # Calculate distinct-n metrics
        metrics = {}

        if all_tokens:
            metrics['distinct-1'] = len(set(all_tokens)) / len(all_tokens)
        else:
            metrics['distinct-1'] = 0.0

        if all_bigrams:
            metrics['distinct-2'] = len(set(all_bigrams)) / len(all_bigrams)
        else:
            metrics['distinct-2'] = 0.0

        if all_trigrams:
            metrics['distinct-3'] = len(set(all_trigrams)) / len(all_trigrams)
        else:
            metrics['distinct-3'] = 0.0

        # Calculate average length
        metrics['avg_length'] = np.mean([len(text.split()) for text in texts])

        return metrics