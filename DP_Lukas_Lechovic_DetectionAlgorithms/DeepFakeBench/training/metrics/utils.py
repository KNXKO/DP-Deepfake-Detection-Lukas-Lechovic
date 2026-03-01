# -*- coding: utf-8 -*-
from sklearn import metrics
import numpy as np

def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str

def get_test_metrics(y_pred, y_true, img_names):
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []

        # Kontrola vstupných parametrov
        if len(image) == 0 or len(pred) == 0 or len(label) == 0:
            print("Warning: Empty input data for video metrics")
            return 0.5, 0.5  # Return default values

        try:
            # Spracovanie ciest k súborom a zoskupenie podľa videí
            for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
                s = item[0]
                if '\\' in s:
                    parts = s.split('\\')
                else:
                    parts = s.split('/')

                # Získanie názvu videa (predposledný element cesty)
                if len(parts) >= 2:
                    a = parts[-2]  # názov videa
                    b = parts[-1]  # názov súboru
                    if a not in result_dict:
                        result_dict[a] = []
                    result_dict[a].append(item)
                else:
                    # Ak cesta nemá správny formát, preskačujeme
                    continue

            # Kontrola či máme nejaké video dáta
            if not result_dict:
                print("Warning: No video data to process")
                return 0.5, 0.5

            # Výpočet priemeru pre každé video
            image_arr = list(result_dict.values())
            for video in image_arr:
                if len(video) == 0:
                    continue

                pred_sum = 0
                label_sum = 0
                leng = 0

                for frame in video:
                    try:
                        pred_sum += float(frame[1])
                        label_sum += int(frame[2])
                        leng += 1
                    except (ValueError, IndexError) as e:
                        print(f"Error processing frame: {e}")
                        continue

                if leng > 0:
                    new_pred.append(pred_sum / leng)
                    new_label.append(int(label_sum / leng))

            # Kontrola či máme dostatok dát pre výpočet metrík
            if len(new_pred) < 2 or len(new_label) < 2:
                print("Warning: Insufficient data for video metrics calculation")
                return 0.5, 0.5

            # Kontrola či máme aspoŘ jednu pozitívnu a jednu negatívnu vzorku
            if len(set(new_label)) < 2:
                print("Warning: All labels are the same")
                return 0.5, 0.5

            # Výpočet ROC krivky
            fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
            v_auc = metrics.auc(fpr, tpr)

            # Výpočet EER s kontrolou NaN hodnôt
            fnr = 1 - tpr
            abs_diff = np.absolute(fnr - fpr)

            # Kontrola či nie sú všetky hodnoty NaN
            if np.all(np.isnan(abs_diff)):
                print("Warning: All values are NaN when calculating EER")
                v_eer = 0.5
            else:
                # Nájdenie indexu s minimálnou absolútnou diferenciou (ignoruje NaN)
                min_idx = np.nanargmin(abs_diff)
                v_eer = fpr[min_idx]

                # Dodatočná kontrola či EER nie je NaN
                if np.isnan(v_eer):
                    v_eer = 0.5

            return v_auc, v_eer

        except Exception as e:
            print(f"Error calculating video metrics: {e}")
            return 0.5, 0.5  # Return default values on error

    y_pred = y_pred.squeeze()
    # Pre UCF, kde labely pre rôzne manipulácie nie sú konzistentné
    y_true[y_true >= 1] = 1

    # Kontrola vstupných dát
    if len(y_pred) == 0 or len(y_true) == 0:
        print("Error: Empty predictions or labels")
        return {'acc': 0.5, 'auc': 0.5, 'eer': 0.5, 'ap': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1_score': 0.5, 'specificity': 0.5, 'pred': y_pred, 'video_auc': 0.5, 'label': y_true}

    try:
        # AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        # EER s kontrolou NaN
        fnr = 1 - tpr
        abs_diff = np.absolute(fnr - fpr)

        if np.all(np.isnan(abs_diff)):
            eer = 0.5
        else:
            min_idx = np.nanargmin(abs_diff)
            eer = fpr[min_idx]
            if np.isnan(eer):
                eer = 0.5

        # AP (Average Precision)
        ap = metrics.average_precision_score(y_true, y_pred)

        # Presnos�
        prediction_class = (y_pred > 0.5).astype(int)
        correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
        acc = correct / len(prediction_class)

        # Doplnkov� metriky
        # Confusion matrix komponenty
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, prediction_class).ravel()

        # Precision (PPV - Positive Predictive Value)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall (Sensitivity, TPR - True Positive Rate)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1-Score (harmonick� priemer precision a recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Specificity (TNR - True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Video-level metrics
        if type(img_names[0]) is not list:
            # Calculate video-level auc for frame-level methods
            v_auc, v_eer = get_video_metrics(img_names, y_pred, y_true)
        else:
            # Video-level methods
            v_auc = auc

        return {
            'acc': acc,
            'auc': auc,
            'eer': eer,
            'ap': ap,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'pred': y_pred,
            'video_auc': v_auc,
            'label': y_true
        }

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'acc': 0.5,
            'auc': 0.5,
            'eer': 0.5,
            'ap': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1_score': 0.5,
            'specificity': 0.5,
            'pred': y_pred,
            'video_auc': 0.5,
            'label': y_true
        }