import pytest
import httpx
from scripts.validation.start_test_server import QAClient


@pytest.mark.asyncio
async def test_request_budget_stops_before_provider_io():
    calls=[]
    class Client:
        async def post(self,*args,**kwargs):calls.append(True);return 'ok'
    client=QAClient(Client(),max_requests=1)
    assert await client.post('https://example.invalid')=='ok'
    with pytest.raises(httpx.RequestError):await client.post('https://example.invalid')
    assert calls==[True] and client.requests==1


@pytest.mark.asyncio
async def test_stream_budget_is_shared_with_non_streamed_calls():
    class Client:
        async def post(self,*args,**kwargs):return 'ok'
    client=QAClient(Client(),max_requests=1)
    await client.post('https://example.invalid')
    with pytest.raises(httpx.RequestError):
        async with client.stream('POST','https://example.invalid'):pass
