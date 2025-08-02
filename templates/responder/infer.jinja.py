{#
 SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#}

    async def {{ name }}(self, request, body, **kwargs):
        accept = request.headers.get("accept", "")
        streaming = "application/x-ndjson" in accept
        try:
            in_data = self.process_request("{{ name }}", body)
            self.logger.debug(f"request processed as {in_data}")
        except Exception as e:
            self.logger.error(f"Request processing failed: {type(e).__name__}: {e}")
            return 400, str(e)

        try:
            if streaming:
                # If streaming, yield results as they are processed
                async def generate_stream():
                    async with self._lock:
                        async for result in self._inference.execute(in_data):
                            response = self.process_response("infer", request, result)
                            yield response

                # Wrap the generator with StreamingResponse
                return 200, StreamingResponse(generate_stream(), media_type="application/x-ndjson")
            else:
                # If not streaming, process and return the last result
                response = None
                async with self._lock:
                    async for result in self._inference.execute(in_data):
                        response = self.process_response("infer", request, result)
                    return (200, response) if response else (500, "No results generated from inference")
        except Exception as e:
            self.logger.error(f"Inference failed: {type(e).__name__}: {e}")
            return 500, str(e)
