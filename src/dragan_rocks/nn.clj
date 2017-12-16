(ns dragan-rocks.nn
  (:require [denisovan.core]
            [clojure.core.matrix :as m]))

(m/set-current-implementation :neanderthal)

(def input-data
 (m/matrix [[0 0 1]
            [0 1 1]
            [1 0 1]
            [1 1 1]]))

(def output-data
  (m/matrix [[0]
             [1]
             [1]
             [0]]))

(defn derivative-sigmoid [x]
  (m/mul x (m/add 1 (m/negate x))))

(defn random-matrix [nrows ncol]
  (let [row (fn [] (->> (repeatedly #(Math/random)) (take ncol)))]
    (m/matrix (repeatedly nrows row))))

(defn synaptic-weight [nrows ncol]
  (->> (random-matrix nrows ncol)
       (m/mmul 2)
       (m/add -1)))

(defn feed-forward [layer weight]
  (-> (m/mmul layer weight) (m/logistic)))

(defn backprop-gradient [layer layer-error]
  (m/mul layer-error (derivative-sigmoid layer)))

(defn update-weights [weights layer next-layer-grad]
  (m/add! weights (m/mmul (m/transpose layer) next-layer-grad)))

(let [out-count (first (m/shape output-data))
      synaptic-weight-0 (synaptic-weight 3 4)
      synaptic-weight-1 (synaptic-weight 4 1)]
  (dotimes [i 60000]
    (let [layer-0 input-data
          layer-1 (feed-forward layer-0 synaptic-weight-0)
          layer-2 (feed-forward layer-1 synaptic-weight-1)
          layer-2-error (m/add output-data (m/negate layer-2))
          layer-2-grad (backprop-gradient layer-2 layer-2-error)
          layer-1-error (m/mmul layer-2-grad (m/transpose synaptic-weight-1))
          layer-1-grad (backprop-gradient layer-1 layer-1-error)
          synaptic-weight-1 (update-weights synaptic-weight-1 layer-1 layer-2-grad)
          synaptic-weight-0 (update-weights synaptic-weight-0 layer-0 layer-1-grad)]
      (when (== 0 (rem i 10000))
        (println (/ (m/ereduce + (m/abs layer-2-error)) out-count))
        (m/pm (m/logistic (m/mmul (m/matrix [0 1 1]) synaptic-weight-0)))))))
