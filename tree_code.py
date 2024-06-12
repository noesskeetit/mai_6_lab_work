import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Находит оптимальный порог для разбиения вектора признака по критерию Джини.

    Критерий Джини определяется следующим образом:
    .. math::
        Q(R) = -\\frac {|R_l|}{|R|}H(R_l) -\\frac {|R_r|}{|R|}H(R_r),

    где:
    * :math:`R` — множество всех объектов,
    * :math:`R_l` и :math:`R_r` — объекты, попавшие в левое и правое поддерево соответственно.

    Функция энтропии :math:`H(R)`:
    .. math::
        H(R) = 1 - p_1^2 - p_0^2,

    где:
    * :math:`p_1` и :math:`p_0` — доля объектов класса 1 и 0 соответственно.

    Указания:
    - Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    - В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака.
    - Поведение функции в случае константного признака может быть любым.
    - При одинаковых приростах Джини нужно выбирать минимальный сплит.
    - Для оптимизации рекомендуется использовать векторизацию вместо циклов.

    Parameters
    ----------
    feature_vector : np.ndarray
        Вектор вещественнозначных значений признака.
    target_vector : np.ndarray
        Вектор классов объектов (0 или 1), длина `feature_vector` равна длине `target_vector`.

    Returns
    -------
    thresholds : np.ndarray
        Отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на
        два различных поддерева.
    ginis : np.ndarray
        Вектор со значениями критерия Джини для каждого порога в `thresholds`.
    threshold_best : float
        Оптимальный порог для разбиения.
    gini_best : float
        Оптимальное значение критерия Джини.

    """
    # ╰( ͡☉ ͜ʖ ͡☉ )つ──☆*:・ﾟ   ฅ^•ﻌ•^ฅ   ʕ•ᴥ•ʔ

    # Сортируем векторы признаков и классов
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    # Находим уникальные значения признаков и их доли
    unique_features = np.unique(sorted_features)

    # Если все значения признака одинаковые, возвращаем любые значения (например, нули)
    if len(unique_features) == 1:
        return np.array([]), np.array([]), None, None

    # Находим все возможные пороги
    thresholds = (unique_features[:-1] + unique_features[1:]) / 2

    # Инициализируем массивы для хранения значений критерия Джини
    ginis = np.zeros(thresholds.shape)

    # Общее количество объектов и доли классов в полном множестве
    total_objects = len(target_vector)
    p1_total = np.sum(sorted_targets) / total_objects
    p0_total = 1 - p1_total

    # Перебираем пороги и вычисляем значения критерия Джини
    for i, threshold in enumerate(thresholds):
        left_mask = sorted_features <= threshold
        right_mask = ~left_mask

        left_size = np.sum(left_mask)
        right_size = total_objects - left_size

        if left_size == 0 or right_size == 0:
            ginis[i] = 1  # Максимально плохой Джини
            continue

        p1_left = np.sum(sorted_targets[left_mask]) / left_size
        p0_left = 1 - p1_left

        p1_right = np.sum(sorted_targets[right_mask]) / right_size
        p0_right = 1 - p1_right

        h_left = 1 - p1_left ** 2 - p0_left ** 2
        h_right = 1 - p1_right ** 2 - p0_right ** 2

        gini = -left_size / total_objects * h_left - right_size / total_objects * h_right
        ginis[i] = gini

    # Находим минимальный Джини и соответствующий порог
    best_index = np.argmin(ginis)
    threshold_best = thresholds[best_index]
    gini_best = ginis[best_index]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        """
        Обучение узла дерева решений.

        Если все элементы в подвыборке принадлежат одному классу, узел становится терминальным.

        Parameters
        ----------
        sub_X : np.ndarray
            Подвыборка признаков.
        sub_y : np.ndarray
            Подвыборка меток классов.
        node : dict
            Узел дерева, который будет заполнен информацией о разбиении.
        depth : int
            Глубина текущего узла.
        """
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, float('inf'), None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {
                    key: clicks.get(key, 0) / count for key, count in counts.items()
                }
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {
                    category: i for i, category in enumerate(sorted_categories)
                }
                feature_vector = np.vectorize(categories_map.get)(sub_X[:, feature])
            else:
                raise ValueError("Некорректный тип признака")

            if len(np.unique(feature_vector)) <= 1:
                continue

            threshold, gini = self.find_best_split(feature_vector, sub_y)

            if gini < gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [
                        k for k, v in categories_map.items() if v < threshold
                    ]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError("Некорректный тип признака")

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание класса для одного объекта по узлу дерева решений.

        Если узел терминальный, возвращается предсказанный класс.
        Если узел не терминальный, выборка передается в соответствующее поддерево для дальнейшего предсказания.

        Parameters
        ----------
        x : np.ndarray
            Вектор признаков одного объекта.
        node : dict
            Узел дерева решений.

        Returns
        -------
        int
            Предсказанный класс объекта.
        """
        if node["type"] == "terminal":
            return node["class"]

        if self._feature_types[node["feature_split"]] == "real":
            if x[node["feature_split"]] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[node["feature_split"]] == "categorical":
            if x[node["feature_split"]] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def find_best_split(self, feature_vector, sub_y):
        # This function should find the best split for the feature_vector
        # Here we use a simple placeholder implementation
        best_threshold = None
        best_gini = float('inf')

        for threshold in np.unique(feature_vector):
            left_mask = feature_vector < threshold
            right_mask = ~left_mask
            gini = self._gini_index(sub_y[left_mask], sub_y[right_mask])

            if gini < best_gini:
                best_gini = gini
                best_threshold = threshold

        return best_threshold, best_gini

    def _gini_index(self, left_y, right_y):
        left_size = len(left_y)
        right_size = len(right_y)
        total_size = left_size + right_size

        if total_size == 0:
            return 0

        left_score = 1.0 - sum((np.sum(left_y == c) / left_size) ** 2 for c in np.unique(left_y))
        right_score = 1.0 - sum((np.sum(right_y == c) / right_size) ** 2 for c in np.unique(right_y))

        gini = (left_size / total_size) * left_score + (right_size / total_size) * right_score
        return gini

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
