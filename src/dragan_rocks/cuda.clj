(ns dragan-rocks.cuda
  (:require [uncomplicate.clojurecuda.core :as core]
            [uncomplicate.clojurecuda.info :as info]
            [uncomplicate.clojurecuda.nvrtc :as nv]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [asum]]
            #_[uncomplicate.neanderthal.cuda :as cuda]))

;; (core/init)

;; (map info/info (map core/device (range (core/device-count))))
