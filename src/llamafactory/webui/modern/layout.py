# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Build-time re-parenting of the containers created by the classic tab factories.

The ``create_*_tab()`` functions build their rows and accordions in a fixed order while binding
events. Gradio only turns the block tree into a page layout when the surrounding ``gr.Blocks``
context exits, and event bindings refer to components by id, not by position. So a container can
be detached from one parent and appended to another before that moment without touching any
callback.

Rules kept here:

* Only whole containers (Row / Column / Form / Accordion) are moved, plus the language dropdown,
  which is re-wrapped in a Form exactly as Gradio itself would do.
* Nothing that still holds a component is ever deleted: ``prune()`` only removes empty containers.
* ``check_reachable()`` proves afterwards that every managed component is still in the tree exactly
  once; the caller aborts the build otherwise.
"""

from collections.abc import Iterable


def _is_context(block) -> bool:
    return hasattr(block, "children") and isinstance(block.children, list)


def parent_of(block, kind: str | None = None):
    r"""Return the nearest ancestor, optionally the nearest one whose class name is ``kind``."""
    node = getattr(block, "parent", None)
    while node is not None and kind is not None and type(node).__name__ != kind:
        node = getattr(node, "parent", None)
    return node


def row_of(block):
    return parent_of(block, "Row")


def detach(block) -> None:
    parent = getattr(block, "parent", None)
    if parent is not None and block in parent.children:
        parent.children.remove(block)
    block.parent = None


def move(block, new_parent, index: int | None = None) -> None:
    if block is None or new_parent is None:
        raise ValueError("move() needs a block and a destination.")

    if not _is_context(new_parent):
        raise TypeError(f"{type(new_parent).__name__} cannot hold children.")

    detach(block)
    if index is None:
        new_parent.children.append(block)
    else:
        new_parent.children.insert(index, block)

    block.parent = new_parent
    if hasattr(new_parent, "page") and hasattr(block, "page"):  # keep multi-page bookkeeping consistent
        block.page = new_parent.page


def move_component_into(component, container) -> None:
    r"""Move a single form component and wrap it in a Form, as Gradio does on context exit."""
    move(component, container)
    container.fill_expected_parents()


def has_component(block) -> bool:
    if not _is_context(block):
        return True

    return any(has_component(child) for child in block.children)


def prune(container) -> int:
    r"""Remove empty descendant containers and return how many were removed."""
    removed = 0
    for child in list(getattr(container, "children", [])):
        if _is_context(child):
            removed += prune(child)
            if not child.children:
                detach(child)
                removed += 1

    return removed


def iter_tree(block):
    yield block
    for child in getattr(block, "children", None) or []:
        yield from iter_tree(child)


def check_reachable(root, components: Iterable) -> list[str]:
    r"""Return the components that are unreachable from ``root`` or appear more than once."""
    seen: dict[int, int] = {}
    for node in iter_tree(root):
        seen[id(node)] = seen.get(id(node), 0) + 1

    problems = []
    for comp in components:
        count = seen.get(id(comp), 0)
        if count == 0:
            problems.append(f"unreachable: {type(comp).__name__} #{getattr(comp, '_id', '?')}")
        elif count > 1:
            problems.append(f"duplicated: {type(comp).__name__} #{getattr(comp, '_id', '?')}")

    return problems


def add_classes(block, *classes: str) -> None:
    existing = getattr(block, "elem_classes", None)
    if existing is None:
        block.elem_classes = list(classes)
        return

    if isinstance(existing, str):
        existing = [existing]
        block.elem_classes = existing

    for cls in classes:
        if cls not in existing:
            existing.append(cls)
