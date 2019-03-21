# pymaws: Matsuno Analytical Wave Solution implemented in Python 3
An analytical solutions set for planetary and inertia gravity waves equations based on the work of [Matsuno(1966)](https://www.jstage.jst.go.jp/article/jmsj1965/44/1/44_1_25/_article). These solutions were succsesfully implemented on the [baroclinic wave test case](https://www.geosci-model-dev-discuss.net/gmd-2018-260/).

## Getting Started

``pymaws`` includes one main function called ``eval_field`` and a dictionary named ``Earth`` containing various parameters needed for the field evaluation. For evaluating field on extra-solar planets see instructions below.

### Installation

``pymaws`` has minimal requirements of:

- Python 3.4 to 3.7
- numpy  1.16

The package can be installed from ``pip``:

  	$ pip install pymaws
    
### Testing
The testing procedure tests all the parts of ``pymaws`` and should not take more than a few seconds.
To run the tests, use python3 from the command line:

	$ python test_pymaws.py
  You should get ``OK`` in the last line.
### Example
The command:

	$ from pymaws import *
imports the function eval_field that calculates the solution for a specific field and the dictionary Earth that contains various planetary parameters needed for the calculation.

	$  {'angular_frequency': 7.29212e-05,
	      'gravitational_acceleration': 9.80616,
	      'mean_radius': 6371220.0,
	      'layer_mean_depth': 30.0}
if you want to use extra-solar parameters, you should create a new dictionary with the SAME keys as shown above and have eval_field call it using the parameters argument.

## Authors

* **Ofer Shamir** - *main version*
* **Shlomi Ziskin Ziv** - *added docstrings, rearanged input/output, packaging and testing*


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

