import random

import re, json


random.seed(1112)

# sample-01: 1112
# sample-02: 2112
# sample-03: 3112
# sample-04: 4112
# sample-05: 5112


def get_semantic_vec(quantifier) -> list:
    concept_name = list(quantifier.keys())[0]

    #print(concept_name)


    totals = [25, 50, 100]
    semantics_vector = []

    for total in totals:
        for num in range(0, total + 1):
            if quantifier[concept_name](total, num):
                semantics_vector.append(1)
            else:
                semantics_vector.append(0)


    #print(len(positive_examples))
    #print(len(negative_examples))

    return semantics_vector


def levenshtein_dist(l1, l2) -> int:
    dist = 0

    for i in range(0, len(l1)):
        if l1[i] != l2[i]:
            dist += 1

    return dist


# q1, q2 are dictionaries such as
# quantifier = {">20": lambda total, num1: num1 > 20}
def find_semantic_dist(q1, q2) -> int:
    vec1 = get_semantic_vec(q1)
    vec2 = get_semantic_vec(q2)

    #print(vec1)
    #print(vec2)
    #print(len(vec1))

    return levenshtein_dist(vec1, vec2)


# concept1 = {"<98 and >0.2": lambda total, num1: (num1 < 98) and (num1 > 1/5 * total)}
# concept2 = {" >0.25": lambda total, num1: num1 > 1/4 * total}
# print(find_semantic_dist(concept1, concept2))


class_1 = []
class_2 = []
class_3 = []
class_4 = []
class_5 = []

original_lens = []

## ----------------- class 1 concepts -------------------------------------
class_1.append({"all": lambda total, num1: num1 == total})
class_1.append({"not_all": lambda total, num1: total > num1})

def make_more(x1):
    return lambda total, num1: num1 > x1

def make_less(x1):
    return lambda total, num1: num1 < x1

def make_equal(x1):
    return lambda total, num1: num1 == x1

def make_not_equal(x1):
    return lambda total, num1: num1 != x1


for x in range(0, 91):
    class_1.append({">" + str(x): make_more(x)})

for x in range(1, 91):
    class_1.append({"<" + str(x): make_less(x)})

# x == 80 -> 21 total positive examples
for x in range(1, 81):
    class_1.append({"=" + str(x): make_equal(x)})
    class_1.append({"!=" + str(x): make_not_equal(x)})


original_lens.append(len(class_1))
## ---------------------------------------------------------------------------


## ----------------- class 2 concepts -------------------------------------
proportions = [1/5, 1/4, 1/3,
               2/5, 1/2, 3/5,
               2/3, 3/4, 4/5]


def make_more_p(pp1):
    return lambda total, num1: num1 > pp1 * total

def make_less_p(pp1):
    return lambda total, num1: num1 < pp1 * total


for p in proportions:
    class_2.append({">" + str(round(p, 2)): make_more_p(p)})
    class_2.append({"<" + str(round(p, 2)): make_less_p(p)})


original_lens.append(len(class_2))
## ---------------------------------------------------------------------------


## ----------------- class 3 concepts -------------------------------------
# (x > c1)  or  (x < c2)
# (x > c1)  and  (x < c2)

def make_or_const(cc1, cc2):
    return lambda total, num1: (num1 > cc1) or (num1 < cc2)

def make_and_const(cc1, cc2):
    return lambda total, num1: (num1 > cc1) and (num1 < cc2)


for c1 in range(5, 95):
    for c2 in range(c1-5, 5, -1):
        class_3.append({">" + str(c1) + " or <" + str(c2): make_or_const(c1, c2)  })


for c1 in range(5, 90):
    for c2 in range(c1+5, 95):
        class_3.append({">" + str(c1) + " and <" + str(c2): make_and_const(c1, c2) })



original_lens.append(len(class_3))

## ----------------- class 4 concepts -------------------------------------
# (x > c)    or    (x < p * n)
# (x > c)    and    (x < p * n)
#
# (x < c)    or    (x > p * n)
# (x < c)    and    (x > p * n)

def make_or_const_p_1(cc1, pp2):
    return lambda total, num1: (num1 > cc1) or (num1 < pp2 * total)

def make_or_const_p_2(cc1, pp2):
    return lambda total, num1: (num1 < cc1) or (num1 > pp2 * total)

def make_and_const_p_1(cc1, pp2):
    return lambda total, num1: (num1 > cc1) and (num1 < pp2 * total)

def make_and_const_p_2(cc1, pp2):
    return lambda total, num1: (num1 < cc1) and (num1 > pp2 * total)


proportions = [1/5, 1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5]

for p in proportions:
    for c in range(5, 95):
        class_4.append({">" + str(c) + " or <" + str(round(p, 2)): make_or_const_p_1(c, p) })
        class_4.append({">" + str(c) + " and <" + str(round(p, 2)): make_and_const_p_1(c, p) })

        class_4.append({"<" + str(c) + " or >" + str(round(p, 2)): make_or_const_p_2(c, p) })
        class_4.append({"<" + str(c) + " and >" + str(round(p, 2)): make_and_const_p_2(c, p) })


original_lens.append(len(class_4))

## ----------------- class 5 concepts -------------------------------------
# (x > p1 * n)    or    (x < p2 * n)
# (x > p1 * n)    and    (x < p2 * n)

# (x > c1) and (x < c2) or (x > c3)
# (x > c1) and (x < c2) or (x < c3)

def make_or_p(pp1, pp2):
    return lambda total, num1: (num1 > pp1 * total) or (num1 < pp2 * total)

def make_and_p(pp1, pp2):
    return lambda total, num1: (num1 > pp1 * total) and (num1 < pp2 * total)


def make_and_or_const_1(cc1, cc2, cc3):
    return lambda total, num1: (num1 > cc1 and num1 < cc2) or (num1 > cc3)


def make_and_or_const_2(cc1, cc2, cc3):
    return lambda total, num1: (num1 > cc1 and num1 < cc2) or (num1 < cc3)


for p1 in proportions:
    for p2 in proportions:
        if p1 > p2:
            class_5.append({">" + str(round(p1, 2)) + " or <" + str(round(p2, 2)): make_or_p(p1, p2) })
        elif p2 > p1:
            class_5.append({">" + str(round(p1, 2)) + " and <" + str(round(p2, 2)): make_and_p(p1, p2) })


for c1 in range(5, 90):
    for c2 in range(c1+5, 95):
        for c3 in range(c2+5, 95):
            class_5.append({">" + str(c1) + " and <" + str(c2) + "or >" + str(c3): make_and_or_const_1(c1, c2, c3) })


for c1 in range(5, 90):
    for c2 in range(c1+5, 95):
        for c3 in range(c1-5, 5, -1):
            class_5.append({">" + str(c1) + " and <" + str(c2) + "or <" + str(c3): make_and_or_const_2(c1, c2, c3) })



original_lens.append(len(class_5))

## ---------------------------------------------------------------------------


def concept_dedup(current_class, prev_class):
    new_class = []

    doc = {}
    for concept in current_class:
        flag = True

        d = []

        for prev_concept in prev_class:
            if find_semantic_dist(concept, prev_concept) < 3:
                flag = False

                prev_name = list(prev_concept.keys())[0]
                d.append(prev_name)
                d.append(str(find_semantic_dist(concept, prev_concept)))
                break


        if flag:
            new_class.append(concept)
        else:
            current_name = list(concept.keys())[0]
            doc[current_name] = d

    return new_class, doc


def intra_class_dedup(this_class):
    new_class = []

    for i in range(0, len(this_class)):
        concept = this_class[i]
        flag = True

        for j in range(i + 1, len(this_class)):
            next_concept = this_class[j]
            if find_semantic_dist(concept, next_concept) < 1:
                flag = False
                break

        if flag:
            new_class.append(concept)

    return new_class

