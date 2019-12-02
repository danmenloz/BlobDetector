for x in range(10):
    for y in range(10):
        print('x: ' + str(x) + ' y: ' + str(y) + ' xy: ' + str(x*y))
        if x*y > 10:
            break
    else:
        print('in else :)')
        continue  # only executed if the inner loop did NOT break
    print('finish')
    break  # only executed if the inner loop DID break
