'''
Created on Apr 6, 2018

@author: animesh
'''
import multiprocessing

POISON_PILL = "STOP"

def process_odds(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]


def main():
    num = multiprocessing.Value('d', 0.0)
    arr = multiprocessing.Array('i', range(100000))
    
    # lastly, create our pool of workers - this spawns the processes, 
    # but they don't start actually doing anything yet
    p = multiprocessing.Process(target=process_odds, args=(num, arr))
    p.start()
    p.join()

    i = 0
    while i > 10:
       p.terminate()
       i += 1


    # now we can check the results
    print(arr[:])

    # ...and exit!
    return


if __name__ == "__main__":
    main()