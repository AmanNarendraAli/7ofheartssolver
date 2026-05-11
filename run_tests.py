import inspect

import test_seven_hearts


def discover_tests() -> list[tuple[str, object]]:
    tests = [
        (name, function)
        for name, function in vars(test_seven_hearts).items()
        if name.startswith("test_") and callable(function)
    ]
    return sorted(tests, key=lambda item: inspect.getsourcelines(item[1])[1])


def main() -> None:
    tests = discover_tests()
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
