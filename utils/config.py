class census:
    """
    Configuration of dataset Census Income
    """

    # the size of total features
    params = 13
    sensitive_param = [9, 1, 8]

    # the valid religion of each feature
    input_bounds = [[1, 9], [0, 7], [0, 39], [0, 15], [0, 6], [0, 13], [0, 5], [0, 4], [0, 1], [0, 99], [0, 39],
                    [0, 99], [0, 39]]

    # the name of each feature
    feature_name = ["age", "workclass", "fnlwgt", "education", "marital_status", "occupation", "relationship", "race",
                    "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]

    sens_name = {9: 'sex', 1: "age", 8: "race"}

    # the name of each class
    class_name = ["low", "high"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


class credit:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    params = 20
    sensitive_param = [9, 13]

    # the valid religion of each feature
    input_bounds = [[0, 3], [1, 80], [0, 4], [0, 10], [1, 200], [0, 4], [0, 4], [1, 4], [0, 1], [0, 2], [1, 4], [0, 3],
                    [1, 8], [0, 2], [0, 2], [1, 4], [0, 3], [1, 2], [0, 1], [0, 1]]

    # the name of each feature
    feature_name = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings_status",
                    "employment", "installment_commitment", "sex", "other_parties",
                    "residence", "property_magnitude", "age", "other_payment_plans", "housing", "existing_credits",
                    "job", "num_dependents", "own_telephone", "foreign_worker"]
    sens_name = {9: 'sex', 13: "age"}

    # the name of each class
    class_name = ["bad", "good"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


class bank:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 16
    sensitive_param = 1

    # the valid religion of each feature
    input_bounds = [[1, 9], [0, 11], [0, 2], [0, 3], [0, 1], [-20, 179], [0, 1], [0, 1], [0, 2], [1, 31], [0, 11],
                    [0, 99], [1, 63], [-1, 39], [0, 1], [0, 3]]

    # the name of each feature
    feature_name = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                    "month", "duration", "campaign", "pdays", "previous", "poutcome"]
    sens_name = {1: 'age'}

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class compas:
    # the size of total features
    params = 14
    sensitive_param = [1, 2, 3]
    # the valid religion of each feature
    input_bounds = [[0, 1], [1, 9], [0, 5], [0, 20], [0, 13], [0, 11], [0, 4], [-6, 10], [0, 90], [0, 2], [-1, 1],
                    [0, 2], [0, 1], [-1, 10]]
    sens_name = {1: 'sex', 2: 'age', 3: 'race'}
    # the name of each feature
    feature_name = ["sex",
                    "age",
                    "race",
                    "juv_fel_count",
                    "juv_misd_count",
                    "juv_other_count",
                    "priors_count",
                    "days_b_screening_arrest",
                    "c_days_from_compas",
                    "c_charge_degree",
                    "is_recid",
                    "r_charge_degree",
                    "is_violent_recid",
                    "v_decile_score"]

    # the name of each class
    class_name = ["Low", "High"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
