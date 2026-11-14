import base64
import re
from pathlib import Path
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

SUBSET_LIST = [
    'Trajectory Reasoning',
    'Action Reasoning',
    'Pointing',
    'State Estimation',
    'Spatial Reasoning',
    'Multi-view Reasoning',
    'Task Reasoning',
    'Other',
]

PROMPT_TEMPLATE = "{question}"
DEFAULT_INSTRUCTIONS = "Which option correctly answers `question`?"
ANSWER_INSTRUCTION = re.compile(r"Please answer directly", re.IGNORECASE)


def parse_options(question: str) -> List[tuple[str, str]]:
    """Parse ``A.`` / ``A:`` / ``A)`` options, including empty option texts."""
    heading = re.search(r"Choices:", question, re.IGNORECASE)
    body = question[heading.end():] if heading else question
    body = ANSWER_INSTRUCTION.split(body, 1)[0].rstrip()
    markers = []
    position = 0
    for expected in range(4):
        letter = chr(ord("A") + expected)
        match = re.search(rf"(?<![A-Za-z0-9]){letter}\s*[.):]\s*", body[position:])
        if match is None:
            break
        markers.append((letter, position + match.start(), position + match.end()))
        position = position + match.end()
    if len(markers) < 2:
        raise ValueError(f"no multiple-choice options found: {question[:120]!r}")
    options = []
    for index, (letter, _, text_start) in enumerate(markers):
        end = markers[index + 1][1] if index + 1 < len(markers) else len(body)
        options.append((letter, body[text_start:end].strip()))
    return options


def interleave(question: str, images: List[Any], visual_indices: List[int]) -> List[Content]:
    """Official eval_harness.py placement: character slices, images first when
    the indices are empty or all zero."""
    if not visual_indices or all(index == 0 for index in visual_indices):
        return [*images, ContentText(text=question)]
    contents: List[Content] = []
    last = 0
    for image, index in sorted(zip(images, visual_indices), key=lambda pair: pair[1]):
        if index == 0:
            contents.append(image)
        elif index <= len(question):
            segment = question[last:index]
            if segment:
                contents.append(ContentText(text=segment))
            contents.append(image)
            last = index
        else:
            contents.append(image)
    if last < len(question):
        contents.append(ContentText(text=question[last:]))
    if not contents:
        contents = [ContentText(text=question), *images]
    return contents


def data_url(path: str) -> str:
    blob = Path(path).read_bytes()
    if blob.startswith(b"\xff\xd8\xff"):
        media = "image/jpeg"
    elif blob.startswith(b"\x89PNG\r\n\x1a\n"):
        media = "image/png"
    elif blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        media = "image/webp"
    else:
        media = "application/octet-stream"
    return f"data:{media};base64,{base64.b64encode(blob).decode('ascii')}"


@register_benchmark(
    BenchmarkMeta(
        name='erqa',
        pretty_name='ERQA',
        tags=[Tags.MULTI_MODAL, Tags.MULTIPLE_CHOICE, Tags.REASONING],
        description="""
## Overview

ERQA (Embodied Reasoning QA) is a benchmark for evaluating spatial reasoning and embodied understanding capabilities of multimodal large language models. It tests models' ability to reason about trajectories, actions, spatial relationships, and task planning in egocentric robotic scenarios.

## Task Description

- **Task Type**: Embodied Spatial Reasoning (Multiple Choice)
- **Input**: Egocentric image(s) + multiple-choice question (A/B/C/D)
- **Output**: Single answer letter (A/B/C/D)
- **Domain**: Robotics, spatial reasoning, embodied AI

## Key Features

- 400 questions across 8 reasoning categories
- Multi-image support (some questions require reasoning across multiple views)
- Categories: Trajectory Reasoning, Action Reasoning, Pointing, State Estimation, Spatial Reasoning, Multi-view Reasoning, Task Reasoning, Other
- Egocentric perspective from robotic manipulation scenarios

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Answers are single letters (A/B/C/D)
""",
        dataset_id='evalscope/ERQA',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        evaluation_version='v1.1',
        supports_choice=True,
    )
)
class ERQAAdapter(VisionLanguageAdapter, MultiChoiceAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record.get('question', '')
        answer = record.get('answer', '')
        question_type = record.get('question_type', 'Unknown')
        options = parse_options(question)
        letters = [letter for letter, _ in options]
        texts = [text for _, text in options]

        media = self._extract_media(record, 'image')
        image_values = [media[index]['url'] for index in sorted(media)]
        visual_indices = [int(value) for value in (record.get('visual_indices') or [])]
        order = (
            sorted(range(len(image_values)), key=lambda index: visual_indices[index])
            if visual_indices else list(range(len(image_values)))
        )
        ordered = [image_values[index] for index in order]
        images = [self._content_image_from_value(value) for value in ordered]
        contents = interleave(question, images, visual_indices)

        internal = {
            'kind': 'choice',
            'state': {'question': question},
            'instructions': DEFAULT_INSTRUCTIONS,
            'criteria': {letter: text for letter, text in options},
            'images': [data_url(value) for value in ordered],
        }
        return Sample(
            input=[ChatMessageUser(content=contents, internal={'systemone': internal})],
            choices=texts,
            target=answer,
            subset_key=question_type,
            metadata={
                'question_id': record.get('question_id'),
                'question_type': question_type,
                'option_letters': letters,
                'image_count': len(ordered),
                'image_order': order,
                'visual_indices': visual_indices,
            },
        )
