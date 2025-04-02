    async def {{ name }}(self, interface, request, body):
        in_data = self.process_request("{{ name }}", body)
        self.logger.debug(f"request processed as {in_data}")
        try:
            result =  await self._inference.execute(in_data)
            response = self.process_response("{{ name }}", request, result)
        except Exception as e:
            return 500, str(e)
        return 200, response
