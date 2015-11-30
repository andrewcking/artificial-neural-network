from neural_network import *
"""
Main
"""
if __name__ == '__main__':
    #Setup
        #2,1 is actually just acting as a single neuron if you think about it
        #I could have made a separate case just for single neurons but for extendability we are just passing it through a layer
    my_network = NeuralNetwork([2,2,1])
    #XOR
    inputs = [[1.0,1.0], [1.0,0.0], [0.0,1.0], [0.0,0.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]
    #From Sample Loan Data
    #inputs = [[20.0,3.5,-5.0], [4.0,4.0, -10.0], [4.0,4.4,-14.1], [10.0,3.0,-10.0], [30.2,0.0,-10.0], [4.0,1.0,-10.0],[4.0,3.2,-25.0],[10.0,0.0,-10.0]]
    #targets = [[1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]

    #Display randomized test with no training
    print("Outputs before training: Randomized")
    #run test algorithm for each set of inputs and print them out
    for i in range(len(targets)):
        print(inputs[i], my_network.run_test(inputs[i]))

    #train
    my_network.train(inputs, targets, 20000)

    #Display test with training
    print("Outputs after training")
    for i in range(len(targets)):
        print(inputs[i], my_network.run_test(inputs[i]))

    while True:
        #Run Test Scenarios
        print("-"*10)
        net_worth = float(input("Net Worth: "))
        mon_salary = float(input("Monthly Salary: "))
        amt_sought = float(input("Amount Sought: "))
        test_array = [net_worth,mon_salary,amt_sought]
        my_network.run_bin_test(test_array)
