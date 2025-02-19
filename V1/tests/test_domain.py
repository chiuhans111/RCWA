from RCWA.Domain import Domain

domain = Domain()
domain.set_period_centered(3, 1)
x, y = domain.get_coordinate(3, 1)
print(x)
print(y)