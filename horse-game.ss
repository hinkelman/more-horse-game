(import (dataframe))


(define rolls-df
  (->
   (dataframe-crossing
    (make-series* (dice1 1 2 3 4 5 6))
    (make-series* (dice2 1 2 3 4 5 6)))
   (dataframe-modify* (roll (dice1 dice2) (+ dice1 dice2)))
   (dataframe-aggregate* (roll) (n (roll) (length roll)))
   (dataframe-modify* (steps () '(3 6 8 11 14 16 14 11 8 6 3)))
   (->> ((lambda (dfx)
	   (-> dfx
	       (dataframe-modify* (total-n () (sum ($ dfx 'n))))
	       (dataframe-modify* (total-steps () (sum ($ dfx 'steps))))))))
   (dataframe-modify*
    (prob (n total-n) (/ n total-n))
    (prob-steps (steps total-steps) (/ steps total-steps)))))

(dataframe-display rolls-df 11)

