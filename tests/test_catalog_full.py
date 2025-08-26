"""Comprehensive tests for catalog.py to achieve >80% coverage."""

import sys
from unittest.mock import patch, MagicMock

import pytest

from mcp_eval.catalog import (
    TaskDefinition,
    TaskRegistry,
    load_module,
    discover_tasks,
    load_task_module,
    register_task,
    get_task,
    get_all_tasks,
    clear_registry,
)


def test_task_definition():
    """Test TaskDefinition dataclass."""
    task = TaskDefinition(
        name="test_task", module="test_module", description="Test task description"
    )
    assert task.name == "test_task"
    assert task.module == "test_module"
    assert task.description == "Test task description"


def test_task_registry_singleton():
    """Test TaskRegistry singleton pattern."""
    registry1 = TaskRegistry()
    registry2 = TaskRegistry()
    assert registry1 is registry2


def test_task_registry_register_and_get():
    """Test registering and retrieving tasks."""
    registry = TaskRegistry()
    registry.clear()  # Clear any existing tasks

    task = TaskDefinition("test", "module", "description")
    registry.register("test_key", task)

    retrieved = registry.get("test_key")
    assert retrieved == task
    assert retrieved.name == "test"


def test_task_registry_get_all():
    """Test getting all tasks."""
    registry = TaskRegistry()
    registry.clear()

    task1 = TaskDefinition("task1", "module1", "desc1")
    task2 = TaskDefinition("task2", "module2", "desc2")

    registry.register("key1", task1)
    registry.register("key2", task2)

    all_tasks = registry.get_all()
    assert len(all_tasks) == 2
    assert "key1" in all_tasks
    assert "key2" in all_tasks


def test_task_registry_clear():
    """Test clearing the registry."""
    registry = TaskRegistry()

    task = TaskDefinition("test", "module", "description")
    registry.register("test", task)

    registry.clear()
    all_tasks = registry.get_all()
    assert len(all_tasks) == 0


def test_register_task():
    """Test the global register_task function."""
    clear_registry()  # Clear before test

    task = TaskDefinition("global_test", "module", "description")
    register_task("global_key", task)

    retrieved = get_task("global_key")
    assert retrieved == task


def test_get_task():
    """Test the global get_task function."""
    clear_registry()

    task = TaskDefinition("get_test", "module", "description")
    register_task("get_key", task)

    retrieved = get_task("get_key")
    assert retrieved.name == "get_test"

    # Test non-existent task
    assert get_task("nonexistent") is None


def test_get_all_tasks():
    """Test the global get_all_tasks function."""
    clear_registry()

    task1 = TaskDefinition("task1", "mod1", "d1")
    task2 = TaskDefinition("task2", "mod2", "d2")

    register_task("t1", task1)
    register_task("t2", task2)

    all_tasks = get_all_tasks()
    assert len(all_tasks) == 2
    assert "t1" in all_tasks
    assert "t2" in all_tasks


def test_clear_registry_global():
    """Test the global clear_registry function."""
    register_task("test", TaskDefinition("t", "m", "d"))
    assert len(get_all_tasks()) > 0

    clear_registry()
    assert len(get_all_tasks()) == 0


def test_load_module_from_file(tmp_path):
    """Test loading a module from a file path."""
    # Create a test module file
    module_file = tmp_path / "test_module.py"
    module_file.write_text("""
def test_function():
    return "hello"

test_var = 42
    """)

    module = load_module(str(module_file))
    assert hasattr(module, "test_function")
    assert hasattr(module, "test_var")
    assert module.test_function() == "hello"
    assert module.test_var == 42


def test_load_module_already_imported():
    """Test loading a module that's already imported."""
    # Import a standard library module
    import json

    # Try to load it again
    module = load_module("json")
    assert module is json


def test_load_module_by_name():
    """Test loading a module by name."""
    module = load_module("os.path")
    assert hasattr(module, "join")
    assert hasattr(module, "exists")


def test_load_module_invalid_file():
    """Test loading from non-existent file."""
    with pytest.raises(ImportError):
        load_module("/nonexistent/module.py")


def test_load_module_invalid_name():
    """Test loading non-existent module by name."""
    with pytest.raises(ImportError):
        load_module("nonexistent_module_12345")


def test_discover_tasks(tmp_path):
    """Test discovering tasks in a module."""
    # Create a test module with task decorators
    module_file = tmp_path / "test_tasks.py"
    module_file.write_text("""
from mcp_eval.core import task

@task(description="First task")
async def first_task(agent, session):
    pass

@task(description="Second task")
async def second_task(agent, session):
    pass

# Not a task
def regular_function():
    pass
    """)

    module = load_module(str(module_file))
    tasks = discover_tasks(module)

    assert len(tasks) == 2
    assert "first_task" in tasks
    assert "second_task" in tasks
    assert "regular_function" not in tasks

    assert tasks["first_task"].description == "First task"
    assert tasks["second_task"].description == "Second task"


def test_discover_tasks_no_tasks(tmp_path):
    """Test discovering tasks in module without tasks."""
    module_file = tmp_path / "no_tasks.py"
    module_file.write_text("""
def function1():
    pass

class MyClass:
    pass
    """)

    module = load_module(str(module_file))
    tasks = discover_tasks(module)
    assert len(tasks) == 0


def test_discover_tasks_with_params(tmp_path):
    """Test discovering parametrized tasks."""
    module_file = tmp_path / "param_tasks.py"
    module_file.write_text("""
from mcp_eval.core import task, parametrize

@parametrize("value", [1, 2, 3])
@task(description="Parametrized task")
async def param_task(agent, session, value):
    pass
    """)

    module = load_module(str(module_file))
    tasks = discover_tasks(module)

    # Should discover the base task
    assert "param_task" in tasks
    assert tasks["param_task"].description == "Parametrized task"


def test_load_task_module_file(tmp_path):
    """Test load_task_module with a file path."""
    clear_registry()

    module_file = tmp_path / "my_tasks.py"
    module_file.write_text("""
from mcp_eval.core import task

@task(description="My test task")
async def my_task(agent, session):
    return "result"
    """)

    load_task_module(str(module_file))

    # Check tasks were registered
    all_tasks = get_all_tasks()
    assert any("my_task" in key for key in all_tasks.keys())


def test_load_task_module_by_name():
    """Test load_task_module with module name."""
    clear_registry()

    # Mock the importlib.import_module to return a fake module
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_module.__name__ = "test_module"

        # Add a mock task
        mock_task = MagicMock()
        mock_task.__name__ = "test_task"
        mock_task.__doc__ = "Test task"
        mock_task._is_mcpeval_task = True
        mock_module.test_task = mock_task

        mock_import.return_value = mock_module

        load_task_module("test_module")

        # Verify import was called
        mock_import.assert_called_once_with("test_module")


def test_load_task_module_invalid():
    """Test load_task_module with invalid input."""
    with pytest.raises(ImportError):
        load_task_module("/nonexistent/file.py")


def test_task_registry_duplicate_register():
    """Test registering duplicate task keys."""
    registry = TaskRegistry()
    registry.clear()

    task1 = TaskDefinition("task1", "module1", "desc1")
    task2 = TaskDefinition("task2", "module2", "desc2")

    registry.register("same_key", task1)
    registry.register("same_key", task2)  # Overwrites

    retrieved = registry.get("same_key")
    assert retrieved.name == "task2"  # Should be the second one


def test_discover_tasks_with_wrapper(tmp_path):
    """Test discovering tasks that have been wrapped."""
    module_file = tmp_path / "wrapped_tasks.py"
    module_file.write_text("""
from mcp_eval.core import task
from functools import wraps

def my_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    wrapper._is_mcpeval_task = True
    wrapper.__doc__ = func.__doc__ or "Wrapped task"
    return wrapper

@my_decorator
@task(description="Decorated task")
async def decorated_task(agent, session):
    pass
    """)

    module = load_module(str(module_file))
    tasks = discover_tasks(module)

    assert "decorated_task" in tasks
    assert tasks["decorated_task"].description == "Decorated task"


def test_load_module_with_syntax_error(tmp_path):
    """Test loading module with syntax error."""
    module_file = tmp_path / "syntax_error.py"
    module_file.write_text("""
def broken_function(
    # Missing closing parenthesis
    """)

    with pytest.raises(ImportError):
        load_module(str(module_file))


def test_discover_tasks_class_methods(tmp_path):
    """Test that class methods are not discovered as tasks."""
    module_file = tmp_path / "class_tasks.py"
    module_file.write_text("""
from mcp_eval.core import task

class TestClass:
    @task(description="Class method task")
    async def class_task(self, agent, session):
        pass

@task(description="Module task")
async def module_task(agent, session):
    pass
    """)

    module = load_module(str(module_file))
    tasks = discover_tasks(module)

    # Should only find module-level tasks
    assert "module_task" in tasks
    assert "class_task" not in tasks


def test_load_module_relative_import(tmp_path):
    """Test loading module with relative path."""
    # Create a package structure
    package_dir = tmp_path / "test_package"
    package_dir.mkdir()

    init_file = package_dir / "__init__.py"
    init_file.write_text("")

    module_file = package_dir / "test_module.py"
    module_file.write_text("""
test_value = "from_package"
    """)

    # Add package parent to path temporarily
    sys.path.insert(0, str(tmp_path))
    try:
        module = load_module("test_package.test_module")
        assert module.test_value == "from_package"
    finally:
        sys.path.pop(0)
