import paper

test = paper.Paper()

test.visualize()
test.fold("north")
print("After folding north:")
test.visualize()
test.fold("east")
print("After folding east:")
test.visualize()