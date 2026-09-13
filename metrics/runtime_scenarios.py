import json
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENARIO_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "runtime_scenarios.json"
)


# =========================================================
# CATALOG
# =========================================================

class RuntimeScenarioCatalog:

    def __init__(
        self,
        config_path=None,
    ):

        if config_path is None:
            config_path = SCENARIO_CONFIG_PATH

        self.config_path = Path(
            config_path
        )

        self.note = ""

        self.scenarios = []

        self.by_key = {}
        self.by_id = {}

        self._load()


    # =====================================================
    # LOAD
    # =====================================================

    def _load(self):

        if not self.config_path.exists():

            raise FileNotFoundError(
                f"Runtime scenario config not found: "
                f"{self.config_path}"
            )


        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)


        self.note = str(
            data.get(
                "prototype_note",
                "",
            )
        )


        scenarios = data.get(
            "scenarios",
            []
        )


        if not isinstance(
            scenarios,
            list,
        ):

            raise ValueError(
                "'scenarios' must be a list."
            )


        if len(scenarios) == 0:

            raise ValueError(
                "No runtime scenarios were defined."
            )


        # -------------------------------------------------
        # VALIDATE
        # -------------------------------------------------

        required_fields = (
            "key",
            "id",
            "title",
            "instruction",
            "validation_target",
            "expected",
        )


        for scenario in scenarios:

            for field in required_fields:

                if field not in scenario:

                    raise ValueError(
                        f"Scenario is missing '{field}': "
                        f"{scenario}"
                    )


            key = str(
                scenario["key"]
            )


            scenario_id = str(
                scenario["id"]
            )


            if key in self.by_key:

                raise ValueError(
                    f"Duplicate scenario key: {key}"
                )


            if scenario_id in self.by_id:

                raise ValueError(
                    f"Duplicate scenario id: {scenario_id}"
                )


            self.scenarios.append(
                scenario
            )


            self.by_key[key] = (
                scenario
            )


            self.by_id[
                scenario_id
            ] = scenario


    # =====================================================
    # ACCESS
    # =====================================================

    def get_by_key(
        self,
        key,
    ):

        return self.by_key.get(
            str(key)
        )


    def get_by_id(
        self,
        scenario_id,
    ):

        return self.by_id.get(
            str(scenario_id)
        )


    def get_default(
        self,
    ):

        return self.scenarios[0]


    # =====================================================
    # SUMMARY
    # =====================================================

    def menu_lines(
        self,
    ):

        lines = []


        for scenario in self.scenarios:

            lines.append(

                (
                    f"[{scenario['key']}] "
                    f"{scenario['id']} - "
                    f"{scenario['title']}"
                )
            )


        return lines


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    print()

    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Runtime Scenario Catalog"
    )

    print(
        "========================================================"
    )


    catalog = (
        RuntimeScenarioCatalog()
    )


    print()

    print(
        "Scenarios loaded:",
        len(
            catalog.scenarios
        )
    )


    print()


    all_pass = True


    expected_ids = [

        "SAFE_ALTERNATIVE",

        "BRIEF_RISK",

        "PERSISTENT_RISK",

        "ORIENTATION_RISK",

        "VIOLATION_RECOVERY",

        "PERSISTENT_VIOLATION",
    ]


    for index, expected_id in enumerate(
        expected_ids,
        start=1,
    ):

        scenario = catalog.get_by_key(
            str(index)
        )


        passed = (

            scenario is not None

            and

            scenario[
                "id"
            ]
            ==
            expected_id
        )


        all_pass = (
            all_pass
            and
            passed
        )


        print(
            f"{index}.",
            scenario[
                "id"
            ]
            if scenario
            else "MISSING"
        )


        if scenario:

            print(
                "   Title:",
                scenario[
                    "title"
                ]
            )


            print(
                "   Goal:",
                scenario[
                    "validation_target"
                ]
            )


        print(
            "   PASS:",
            passed
        )


        print()


    # =====================================================
    # DEFAULT
    # =====================================================

    default_scenario = (
        catalog.get_default()
    )


    default_pass = (

        default_scenario[
            "id"
        ]
        ==
        "SAFE_ALTERNATIVE"
    )


    all_pass = (
        all_pass
        and
        default_pass
    )


    print(
        "Default scenario:",
        default_scenario[
            "id"
        ]
    )


    print(
        "Default PASS:",
        default_pass
    )


    # =====================================================
    # UNKNOWN KEY
    # =====================================================

    unknown_pass = (

        catalog.get_by_key(
            "9"
        )
        is None
    )


    all_pass = (
        all_pass
        and
        unknown_pass
    )


    print()

    print(
        "Unknown key returns None:",
        unknown_pass
    )


    # =====================================================
    # FINAL
    # =====================================================

    print()

    print(
        "========================================================"
    )


    print(
        "PASS:",
        all_pass
    )


    if all_pass:

        print(
            "RUNTIME SCENARIO CATALOG: PASSED"
        )

    else:

        print(
            "RUNTIME SCENARIO CATALOG: FAILED"
        )


    print(
        "========================================================"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        catalog.note
    )

    print()